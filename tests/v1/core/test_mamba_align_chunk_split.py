# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefill chunk splitting (`_mamba_block_aligned_split`).

Invariant: slot `p` holds the state after exactly `(p + 1) * block_size` tokens.
State is written at chunk ends, so a chunk ending mid-block leaves its slot at
the wrong offset, and a later chunk crossing that boundary publishes it anyway.
Requests resuming from it then restore a truncated state (#43559).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from .utils import create_requests

pytestmark = pytest.mark.cpu_test

# Mirrors the deployment where the poisoning was observed (Qwen3.6-27B): mamba
# block 1600, MTP with 3 draft tokens, prompts shorter than 2 mamba blocks.
ATTN_BLOCK_SIZE = 16
MAMBA_BLOCK_SIZE = 1600
NUM_SPEC = 3
PROMPT_LEN = 2002
MAMBA_GROUP_ID = 1


def _make_hybrid_kv_cache_manager(
    num_prefill_checkpoint_blocks: int = 0,
) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=ATTN_BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=MAMBA_BLOCK_SIZE,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=NUM_SPEC,
                    num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
                    prefill_checkpoint_alignment=(
                        16 if num_prefill_checkpoint_blocks else None
                    ),
                ),
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=262144,
        scheduler_block_size=MAMBA_BLOCK_SIZE,
        hash_block_size=ATTN_BLOCK_SIZE,
        enable_caching=True,
        use_eagle=True,
    )


def _split(
    request: Request,
    num_new_tokens: int,
    use_eagle: bool = True,
    partial_hit: bool = False,
    num_prefill_checkpoint_blocks: int = 0,
    dflash_skip_backoff: bool = False,
    draft_replay_reserve: int = 0,
) -> int:
    """Call the real `Scheduler._mamba_block_aligned_split` on a stub self."""
    block_drop = use_eagle and not dflash_skip_backoff

    def get_replay_boundary(req: Request) -> int:
        max_replay = max(
            req.num_prompt_tokens - 1 - draft_replay_reserve,
            0,
        )
        return max_replay // MAMBA_BLOCK_SIZE * MAMBA_BLOCK_SIZE

    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE),
        use_eagle=use_eagle,
        use_eagle_block_drop=block_drop,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        # `prefix_match_unit` finer than the block size (#46384).
        mamba_partial_cache_hit=partial_hit,
        mamba_fine_grained_prefix_cache=False,
        hash_block_size=ATTN_BLOCK_SIZE,
        mamba_has_prefill_checkpoint_blocks=(
            num_prefill_checkpoint_blocks > 0 and not use_eagle
        ),
        draft_replay_reserve=draft_replay_reserve,
        kv_cache_manager=SimpleNamespace(get_replay_boundary=get_replay_boundary),
    )
    return Scheduler._mamba_block_aligned_split(stub, request, num_new_tokens)


@pytest.mark.parametrize(
    ("prompt_len", "num_new_tokens", "use_eagle", "expected"),
    [
        (2002, 2002, False, 2002),
        (3602, 2000, False, 2000),
        (3602, 3602, True, MAMBA_BLOCK_SIZE),
    ],
)
def test_internal_checkpoint_split(
    prompt_len: int, num_new_tokens: int, use_eagle: bool, expected: int
) -> None:
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    assert (
        _split(
            request,
            num_new_tokens,
            use_eagle=use_eagle,
            num_prefill_checkpoint_blocks=1,
        )
        == expected
    )
    if not use_eagle:
        manager = _make_hybrid_kv_cache_manager(num_prefill_checkpoint_blocks=1)
        assert manager.allocate_slots(request, expected) is not None
        mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
        blocks = mamba_manager.req_to_blocks[request.request_id]
        assert all(not block.is_null for block in blocks)  # checkpoint + running state


def test_dflash_noncacheable_draft_keeps_last_mamba_boundary() -> None:
    (request,) = create_requests(1, num_tokens=3602, block_size=ATTN_BLOCK_SIZE)
    assert _split(request, 3602, dflash_skip_backoff=True) == 2 * MAMBA_BLOCK_SIZE


def test_dflash_does_not_shift_partial_tail_boundary() -> None:
    """DFlash owns its draft KV and must not use EAGLE's target-KV drop."""
    (request,) = create_requests(1, num_tokens=PROMPT_LEN, block_size=ATTN_BLOCK_SIZE)
    without_drop = _split(
        request,
        PROMPT_LEN,
        partial_hit=True,
        dflash_skip_backoff=True,
    )
    with_drop = _split(request, PROMPT_LEN, partial_hit=True)
    assert without_drop == MAMBA_BLOCK_SIZE
    expected_eagle_tail = PROMPT_LEN // ATTN_BLOCK_SIZE * ATTN_BLOCK_SIZE
    assert with_drop == expected_eagle_tail - ATTN_BLOCK_SIZE


def test_dflash_replay_boundary_is_a_real_chunk_end() -> None:
    """The future cache-hit cap must name state written by this producer."""
    prompt_len = 92377
    replay_reserve = 2048
    start = 78336
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    request.num_computed_tokens = start

    scheduled = _split(
        request,
        prompt_len - start,
        dflash_skip_backoff=True,
        draft_replay_reserve=replay_reserve,
    )
    replay_boundary = (
        (prompt_len - 1 - replay_reserve)
        // MAMBA_BLOCK_SIZE
        * MAMBA_BLOCK_SIZE
    )
    assert start + scheduled == replay_boundary


@pytest.mark.parametrize(
    ("main_tokens", "expected_blocks"),
    [
        (MAMBA_BLOCK_SIZE - 1, 1 + NUM_SPEC),
        (MAMBA_BLOCK_SIZE, 2 + NUM_SPEC),
    ],
)
def test_align_lookahead_materializes_exactly_one_boundary_page(
    main_tokens: int, expected_blocks: int
) -> None:
    """Estimator and allocator agree, and only a page boundary gains a page."""
    manager = _make_hybrid_kv_cache_manager()
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    req_id = f"lookahead-{main_tokens}"
    tokens_with_lookahead = main_tokens + 1

    estimated = mamba_manager.get_num_blocks_to_allocate(
        req_id,
        tokens_with_lookahead,
        [],
        total_computed_tokens=0,
        num_local_computed_tokens=0,
        num_tokens_main_model=main_tokens,
    )
    allocated = mamba_manager.allocate_new_blocks(
        req_id, tokens_with_lookahead, main_tokens
    )

    assert estimated == expected_blocks
    assert len(allocated) == estimated
    assert len(mamba_manager.req_to_blocks[req_id]) == expected_blocks
    assert all(not block.is_null for block in mamba_manager.req_to_blocks[req_id])


def test_align_running_request_estimates_one_page_at_next_boundary() -> None:
    manager = _make_hybrid_kv_cache_manager()
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    req_id = "lookahead-running"

    first = mamba_manager.allocate_new_blocks(
        req_id, MAMBA_BLOCK_SIZE + 1, MAMBA_BLOCK_SIZE
    )
    assert len(first) == 2 + NUM_SPEC

    main_tokens = 2 * MAMBA_BLOCK_SIZE
    estimated = mamba_manager.get_num_blocks_to_allocate(
        req_id,
        main_tokens + 1,
        [],
        total_computed_tokens=MAMBA_BLOCK_SIZE,
        num_local_computed_tokens=MAMBA_BLOCK_SIZE,
        num_tokens_main_model=main_tokens,
    )
    allocated = mamba_manager.allocate_new_blocks(
        req_id, main_tokens + 1, main_tokens
    )

    assert estimated == 1
    assert len(allocated) == estimated


def test_align_lookahead_does_not_move_internal_checkpoint() -> None:
    main_tokens = 3 * MAMBA_BLOCK_SIZE

    def checkpoint_for(num_tokens: int) -> tuple[int, int]:
        manager = _make_hybrid_kv_cache_manager(num_prefill_checkpoint_blocks=1)
        mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
        req_id = f"checkpoint-{num_tokens}"
        mamba_manager.get_num_blocks_to_allocate(
            req_id,
            num_tokens,
            [],
            total_computed_tokens=MAMBA_BLOCK_SIZE,
            num_local_computed_tokens=MAMBA_BLOCK_SIZE,
            num_tokens_main_model=main_tokens,
        )
        return mamba_manager._checkpoints[req_id]

    assert checkpoint_for(main_tokens + 1) == checkpoint_for(main_tokens)


def _run_chunked_prefill(
    manager: KVCacheManager, request: Request, budgets: list[int]
) -> dict[int, int]:
    """Prefill `request`, one step per entry in `budgets`.

    A zero-token split means the budget cannot fund an aligned chunk; the
    scheduler defers the request to a later step, so this does too.

    Returns physical mamba block id -> the token offset of the state it holds,
    mirroring the GDN kernel: the running slot ends up at the chunk end.
    """
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    state_at: dict[int, int] = {}
    # `budgets` fragments the first steps; afterwards the request is alone and
    # gets as much as it can use, so the prefill always finishes.
    for step in range(len(budgets) + 64):
        computed = request.num_computed_tokens
        if computed >= request.num_tokens:
            break
        budget = budgets[step] if step < len(budgets) else request.num_tokens
        num_new = _split(request, min(request.num_tokens - computed, budget))
        if num_new == 0:
            continue
        assert (
            manager.allocate_slots(request, num_new, num_lookahead_tokens=NUM_SPEC)
            is not None
        )
        request.num_computed_tokens = computed + num_new
        blocks = mamba_manager.req_to_blocks[request.request_id]
        running = cdiv(request.num_computed_tokens, MAMBA_BLOCK_SIZE) - 1
        state_at[blocks[running].block_id] = request.num_computed_tokens
    return state_at


def _count_cached_boundary_states(
    manager: KVCacheManager, request: Request, state_at: dict[int, int]
) -> int:
    """Assert every hash-cached mamba slot holds the state its hash claims.

    Covers both full-block snapshots (`(p + 1) * block_size`) and the
    partial-tail entries align mode registers at an exact token count.

    Returns the number of cached slots checked.
    """
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    checked = 0
    for pos, block in enumerate(mamba_manager.req_to_blocks[request.request_id]):
        if block.is_null or block.block_hash is None:
            continue
        claimed = block.block_hash_num_tokens
        assert state_at.get(block.block_id) == claimed, (
            f"mamba slot {pos} is hashed as state@{claimed} but holds "
            f"state@{state_at.get(block.block_id)}"
        )
        checked += 1
    return checked


def _prefill(prompt_len: int, budgets: list[int]) -> int:
    manager = _make_hybrid_kv_cache_manager()
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    state_at = _run_chunked_prefill(manager, request, budgets)
    assert request.num_computed_tokens == prompt_len, "prefill did not complete"
    return _count_cached_boundary_states(manager, request, state_at)


def test_fragmented_first_chunk_does_not_poison_mamba_prefix_cache() -> None:
    """EAGLE zeroes `last_cache_position` for prompts under two blocks.

    Past that point any chunk end used to be accepted, so a short first chunk
    (concurrent prefills sharing the budget) left slot 0 at state@364 while the
    next chunk crossed 1600 and published slot 0 as state@1600.
    """
    _prefill(PROMPT_LEN, budgets=[364, PROMPT_LEN])


def test_fragmented_tail_chunk_does_not_poison_mamba_prefix_cache() -> None:
    """Same poisoning one block in, where a hit is still cacheable.

    `last_cache_position` is 1600, so the chunk ending there is cached. The next
    chunk used to be free to stop mid-block (slot 1 at state@2600) and the one
    after it crossed 3200, publishing slot 1 as state@3200.
    """
    assert _prefill(3602, budgets=[1600, 1000, 3602]) > 0


@pytest.mark.parametrize("first_chunk", [800, 900, 1599, 1601, 2000])
def test_intermediate_chunk_ends_stay_block_aligned(first_chunk: int) -> None:
    """Every non-final prefill chunk must end on a mamba block boundary."""
    _prefill(PROMPT_LEN, budgets=[first_chunk, PROMPT_LEN, PROMPT_LEN])


@pytest.mark.parametrize(
    ("block_size", "prompt_len", "budgets"),
    [
        # Kimi-K3-scale mamba blocks: TP8 shards the recurrent state 8 ways
        # (~1.5k block), DEP16 keeps it whole (~12k block). The first budget
        # walks the request up to `last_cache_position`; the second is the
        # sub-block fragment that lands in the unguarded tail region.
        (1536, 30000, [27648, 1024, 30000]),
        (12288, 30000, [12288, 4000, 30000]),
        (12288, 41000, [24576, 8000, 41000]),
    ],
)
def test_poisoning_is_block_size_independent(
    monkeypatch: pytest.MonkeyPatch,
    block_size: int,
    prompt_len: int,
    budgets: list[int],
) -> None:
    """The invariant is per-block, so large mamba blocks are not safer."""
    import sys

    monkeypatch.setattr(sys.modules[__name__], "MAMBA_BLOCK_SIZE", block_size)
    assert _prefill(prompt_len, budgets=budgets) > 0


@pytest.mark.parametrize("partial_hit", [False, True])
@pytest.mark.parametrize("resume_at", [331, 1599, 1601, 2531, 3011])
@pytest.mark.parametrize(
    ("num_prefill_checkpoint_blocks", "use_eagle"),
    [(0, True), (1, False)],
)
def test_unaligned_resume_never_runs_past_its_block(
    partial_hit: bool,
    resume_at: int,
    num_prefill_checkpoint_blocks: int,
    use_eagle: bool,
) -> None:
    """A prefill resuming mid-block must re-align before crossing a boundary.

    Reachable with a finer `prefix_match_unit` (its partial-tail stop ends a
    chunk off-grid by design) and with unaligned external tokens from a KV
    connector.
    """
    prompt_len = 3602
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    # Under eagle the partial-tail stop sits one hash unit below the prompt's
    # last hash boundary: eagle matches a unit past its candidate and drops it,
    # so nothing proves that last boundary.
    tail_stop = prompt_len // ATTN_BLOCK_SIZE * ATTN_BLOCK_SIZE
    if use_eagle:
        tail_stop -= ATTN_BLOCK_SIZE

    pos, ends = resume_at, []
    while pos < prompt_len:
        request.num_computed_tokens = pos
        num_new = _split(
            request,
            prompt_len - pos,
            use_eagle=use_eagle,
            partial_hit=partial_hit,
            num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
        )
        assert num_new > 0, f"no progress at {pos}"
        if pos % MAMBA_BLOCK_SIZE != 0:
            block_end = (pos // MAMBA_BLOCK_SIZE + 1) * MAMBA_BLOCK_SIZE
            assert pos + num_new <= block_end, (
                f"chunk [{pos}, {pos + num_new}) starts mid-block and runs past "
                f"{block_end}; the slot holding state@{pos} gets hashed as "
                f"state@{block_end}"
            )
        pos += num_new
        ends.append(pos)

    for end in ends[:-1]:
        aligned = end % MAMBA_BLOCK_SIZE == 0
        assert aligned or (partial_hit and end == tail_stop), (
            f"intermediate chunk end {end} is neither block-aligned nor the "
            f"partial-tail stop ({tail_stop})"
        )


def test_align_retention_reages_early_freed_boundary_blocks():
    """Boundary-state blocks freed mid-request are re-aged at retirement.

    Align mode frees a request's older boundary states while it is still
    running, leaving them at the LRU-old end of the free queue; the request's
    attention prefix is freed minutes later at the young end, so under churn
    the state is recycled first and the joint hit collapses to zero. At
    retirement the surviving states must move behind blocks freed in between,
    latest boundary first.
    """
    manager = _make_hybrid_kv_cache_manager()
    pool = manager.block_pool
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    req_id = "req-retention"

    boundary1, boundary2, current = pool.get_new_blocks(3)
    h1 = make_block_hash_with_group_id(BlockHash(b"b1"), MAMBA_GROUP_ID)
    h2 = make_block_hash_with_group_id(BlockHash(b"b2"), MAMBA_GROUP_ID)
    boundary1.set_block_hash(h1, MAMBA_BLOCK_SIZE)
    boundary2.set_block_hash(h2, 2 * MAMBA_BLOCK_SIZE)
    pool.cached_block_hash_to_block.insert(h1, boundary1)
    pool.cached_block_hash_to_block.insert(h2, boundary2)

    mamba_manager.req_to_blocks[req_id] = [boundary1, boundary2, current]
    mamba_manager._allocated_block_reqs.add(req_id)
    mamba_manager.last_state_block_idx[req_id] = 0

    # Advancing past the third block early-frees both boundary blocks; they
    # stay hash-cached and must be tracked for retirement re-aging.
    mamba_manager.remove_skipped_blocks(req_id, 3 * MAMBA_BLOCK_SIZE)
    assert boundary1.ref_cnt == 0 and boundary2.ref_cnt == 0
    assert [b for b, _, _ in mamba_manager._freed_boundary_blocks[req_id]] == [
        boundary2,
        boundary1,
    ]

    # Traffic frees another cached block afterwards: without re-aging it
    # would outlive both boundary states.
    (sentinel,) = pool.get_new_blocks(1)
    h3 = make_block_hash_with_group_id(BlockHash(b"sentinel"), 0)
    sentinel.set_block_hash(h3, ATTN_BLOCK_SIZE)
    pool.cached_block_hash_to_block.insert(h3, sentinel)
    pool.free_blocks([sentinel])

    manager.free(SimpleNamespace(request_id=req_id))

    assert req_id not in mamba_manager._freed_boundary_blocks
    order = pool.free_block_queue.get_all_free_blocks()
    # Latest boundary evicted first among the re-aged states, both behind the
    # sentinel; the uncached running block is prepended for first eviction.
    assert order[-2:] == [boundary2, boundary1]
    assert order.index(sentinel) < order.index(boundary2)
    assert order[0] is current


def test_align_retention_skips_recycled_boundary_blocks():
    """A tracked boundary block recycled before retirement is not re-aged."""
    manager = _make_hybrid_kv_cache_manager()
    pool = manager.block_pool
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    req_id = "req-recycled"

    boundary, current = pool.get_new_blocks(2)
    h1 = make_block_hash_with_group_id(BlockHash(b"b1"), MAMBA_GROUP_ID)
    boundary.set_block_hash(h1, MAMBA_BLOCK_SIZE)
    pool.cached_block_hash_to_block.insert(h1, boundary)

    mamba_manager.req_to_blocks[req_id] = [boundary, current]
    mamba_manager._allocated_block_reqs.add(req_id)
    mamba_manager.last_state_block_idx[req_id] = 0

    mamba_manager.remove_skipped_blocks(req_id, 2 * MAMBA_BLOCK_SIZE)
    assert len(mamba_manager._freed_boundary_blocks[req_id]) == 1

    # Later traffic frees a cached sentinel, then the tracked block is
    # recycled (hash dropped). Without the identity guard, retirement would
    # move the recycled block past the sentinel.
    (sentinel,) = pool.get_new_blocks(1)
    h3 = make_block_hash_with_group_id(BlockHash(b"sentinel"), 0)
    sentinel.set_block_hash(h3, ATTN_BLOCK_SIZE)
    pool.cached_block_hash_to_block.insert(h3, sentinel)
    pool.free_blocks([sentinel])
    pool._maybe_evict_cached_block(boundary)
    assert boundary.block_hash is None

    manager.free(SimpleNamespace(request_id=req_id))

    assert req_id not in mamba_manager._freed_boundary_blocks
    order = pool.free_block_queue.get_all_free_blocks()
    assert order.index(boundary) < order.index(sentinel)
