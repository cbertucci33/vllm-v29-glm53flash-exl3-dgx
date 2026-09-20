# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv, _score_edges
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


def _stub_base(monkeypatch, draft_logits):
    """A DFlashSpeculator.__init__ that allocates only what the base class would.

    The real base class fills draft_logits from draft_logits_spec, so callers
    pass a tensor already in that state.
    """

    def init_base(self, _vllm_config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 3})
        )
        self.speculative_config = SimpleNamespace(
            uses_dynamic_speculative_decoding=lambda: False
        )
        self.max_num_reqs = 2
        self.num_query_per_req = 5
        self.num_speculative_steps = 4
        self.vocab_size = 17
        self.draft_tokens = torch.empty((2, 4), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)


def test_selector_refuses_dynamic_speculative_depths(monkeypatch):
    """Per-depth CUDA-graph buckets would trip the fixed-depth walk mid-boot."""

    def init_base(self, _vllm_config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 3})
        )
        self.speculative_config = SimpleNamespace(
            uses_dynamic_speculative_decoding=lambda: True
        )

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)
    with pytest.raises(ValueError, match="fixed physical depth"):
        DFlash2Speculator(None, torch.device("cpu"))


def test_selector_leaves_greedy_drafting_without_proposal_logits(monkeypatch):
    """Greedy is the default, and it caches no proposal distribution.

    The base class allocates draft_logits only for "probabilistic"; verification
    reads `draft_logits is None` to decide whether a distribution is on offer, so
    allocating one here would claim a proposal the walk never sampled from.
    """
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))

    assert speculator.draft_logits is None


def test_top_k_shard_mapping_honors_added_vocab_regions():
    """Shard layout is [org | org_pad | added | added_pad]; the k-wide top-k
    must mask both pad slabs and map added-vocab winners to their global ids,
    exactly as get_top_tokens does."""
    from vllm.model_executor.layers.logits_processor import (
        _globalize_token_ids,
        _mask_vocab_padding,
    )

    shard = SimpleNamespace(
        num_org_elements=6,
        num_org_elements_padded=8,
        num_org_vocab_padding=2,
        num_added_elements=3,
        num_added_elements_padded=4,
        num_added_vocab_padding=1,
        org_vocab_start_index=100,
        added_vocab_start_index=1000,
    )
    logits = torch.zeros(2, 12)
    _mask_vocab_padding(logits, shard)
    assert torch.isinf(logits[:, 6:8]).all() and (logits[:, 6:8] < 0).all()
    assert torch.isinf(logits[:, 11:12]).all()
    assert torch.isfinite(logits[:, :6]).all()
    assert torch.isfinite(logits[:, 8:11]).all()

    ids = torch.tensor([0, 5, 8, 10], dtype=torch.int64)
    torch.testing.assert_close(
        _globalize_token_ids(ids, shard),
        torch.tensor([100, 105, 1000, 1002], dtype=torch.int64),
    )


def test_selector_asks_for_fp32_proposal_logits():
    """The spec the base class allocates from: fp32, filled -inf.

    Not the head dtype -- rounding selector scores to bf16 moves the argmax of a
    candidate row often enough that the walk and the rejection sampler checking it
    would no longer read the same distribution.
    """
    dtype, fill = DFlash2Speculator.draft_logits_spec(None, None)

    assert dtype is torch.float32
    assert fill == float("-inf")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ring_synthesis_covers_context_and_draft_queries():
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        synthesize_draft_ring_block_tables,
    )

    ring_size = 4
    block_table = torch.tensor(
        [[0, 0, 17, 23, 9, 41, 0], [5, 6, 7, 0, 0, 0, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([20, 12], dtype=torch.int32, device="cuda")
    synthesize_draft_ring_block_tables(
        block_table,
        idx_mapping,
        seq_lens,
        block_size=4,
        ring_size=ring_size,
        num_query_per_req=5,
    )

    base0, base1 = 1 + 3 * ring_size, 1 + ring_size
    expected = torch.tensor(
        [
            [base0, base0 + 1, base0 + 2, base0 + 3, base0, base0 + 1, base0 + 2],
            [base1, base1 + 1, base1 + 2, base1 + 3, base1, 0, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    assert torch.equal(block_table, expected)


@pytest.mark.skip_global_cleanup
def test_dflash2_model_decoder_layer_cls(monkeypatch):
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.qwen3_dflash2 import (
        DFlash2Qwen3DecoderLayer,
        DFlash2Qwen3Model,
    )

    # 1. Mock get_current_vllm_config and TP groups
    mock_current_vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
            kv_cache_dtype_skip_layers=[],
            cache_dtype="auto",
            sliding_window=None,
            enable_prefix_caching=False,
        ),
        kv_transfer_config=None,
        speculative_config=None,
        attention_config=SimpleNamespace(
            use_non_causal=False,
            backend=None,
            backend_per_kind={},
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            compile_custom_ops=False,
            custom_ops="all",
            enabled_custom_ops=set(),
            static_forward_context={},
            mode=0,  # CompilationMode.NONE is 0
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
        ),
        kernel_config=SimpleNamespace(
            linear_backend="auto",
        ),
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform,
        "get_attn_backend_cls",
        lambda *args, **kwargs: (
            "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
        ),
    )

    class MockGroup:
        rank_in_group = 0
        world_size = 1

    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        MockGroup(),
    )
    # 2. Mock vllm_config
    hf_config = SimpleNamespace(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_parameters={},
        intermediate_size=512,
        hidden_act="silu",
        dflash_config={
            "selector_rank": 4,
            "selector_top_k": 3,
            "conv_kernel_size": 3,
            "conv_group_size": 2,
            "block_size": 8,
            "use_aux_hidden_state": False,
        },
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=hf_config,
                quantization=None,
            ),
            num_speculative_tokens=5,
            enable_adaptive_verification=False,
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
        ),
        load_config=SimpleNamespace(
            quantization=None,
            quantization_param_path=None,
        ),
    )
    mock_current_vllm_config.speculative_config = vllm_config.speculative_config
    vllm_config.compilation_config = mock_current_vllm_config.compilation_config

    # 3. Instantiate the model under meta device to avoid parameter allocation issues
    with set_current_vllm_config(mock_current_vllm_config), torch.device("meta"):
        model = DFlash2Qwen3Model(vllm_config=vllm_config)

    # 4. Assert that the layers are DFlash2Qwen3DecoderLayer (the subclass)
    assert len(model.layers) == 2
    assert isinstance(model.layers[0], DFlash2Qwen3DecoderLayer)
    assert model.layers[0].attention_conv.block_size == 6


def _fake_dflash_model_and_attn(weight: torch.Tensor, q_size: int, scales=None):
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    qkv_proj = SimpleNamespace(weight=weight)
    if scales is not None:
        qkv_proj.weight_scale = scales
    fake_model = SimpleNamespace(
        hidden_norm=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))
    )
    attn = SimpleNamespace(qkv_proj=qkv_proj, q_size=q_size)
    return DFlashQwen3Model._kv_projection_rows, fake_model, attn


def test_kv_projection_rows_passthrough_for_plain_weights():
    weight = torch.randn(12, 64, dtype=torch.bfloat16)
    rows_fn, fake_model, attn = _fake_dflash_model_and_attn(weight, q_size=8)
    torch.testing.assert_close(rows_fn(fake_model, attn), weight[8:])


def test_kv_projection_rows_dequantizes_mxfp8():
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
    )

    torch.manual_seed(2)
    weight = torch.randn(12, 64).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 134, (12, 2), dtype=torch.uint8)
    rows_fn, fake_model, attn = _fake_dflash_model_and_attn(
        weight, q_size=8, scales=scales
    )
    expected = dequant_mxfp8_to_bf16(weight[8:].contiguous(), scales[8:].contiguous())
    torch.testing.assert_close(rows_fn(fake_model, attn), expected)


def test_kv_projection_rows_rejects_fp8_without_mxfp8_scales():
    weight = torch.randn(12, 64).to(torch.float8_e4m3fn)
    rows_fn, fake_model, attn = _fake_dflash_model_and_attn(weight, q_size=8)
    with pytest.raises(ValueError, match="weight_scale"):
        rows_fn(fake_model, attn)
