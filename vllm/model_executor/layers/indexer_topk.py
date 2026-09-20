# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared NVIDIA sparse-indexer top-k dispatcher."""

import torch

from vllm.platforms import current_platform
from vllm.v1.worker.workspace import current_workspace_manager

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
AUTO_COOPERATIVE_MAX_ROWS = 64
_NATIVE_TOPK_VALUES = (512, 1024, 2048)


def _cooperative_failures(
    logits: torch.Tensor, topk_tokens: int, num_rows: int
) -> list[str]:
    failures: list[str] = []
    if not current_platform.is_cuda():
        failures.append("CUDA is required")
    if topk_tokens not in _NATIVE_TOPK_VALUES:
        failures.append(f"topk_tokens must be in {_NATIVE_TOPK_VALUES}")
    if num_rows > AUTO_COOPERATIVE_MAX_ROWS:
        failures.append(f"num_rows must be <= {AUTO_COOPERATIVE_MAX_ROWS}")
    if logits.stride(0) % 4 != 0:
        failures.append("logits row stride must be 16-byte aligned")
    if not current_platform.has_device_capability(90):
        failures.append("compute capability >= 9.0 is required")
    if current_platform.is_device_capability_family(120):
        failures.append("cooperative_topk is not supported on SM12x")
    return failures


def run_indexer_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    next_n: int,
    output: torch.Tensor,
    topk_tokens: int,
    max_seq_len: int,
    backend: str,
) -> None:
    """Select sparse-indexer top-k with one fail-closed backend contract."""
    num_rows = logits.shape[0]
    selected = backend
    cooperative_failures = _cooperative_failures(logits, topk_tokens, num_rows)
    if selected == "auto":
        if not cooperative_failures:
            selected = "cooperative"
        elif current_platform.is_cuda() and topk_tokens in _NATIVE_TOPK_VALUES:
            selected = "persistent"
        else:
            selected = "per_row"

    if selected == "cooperative":
        if cooperative_failures:
            raise ValueError(
                "cooperative sparse-indexer top-k is unavailable: "
                + "; ".join(cooperative_failures)
            )
        (workspace,) = current_workspace_manager().get_simultaneous(
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )
        torch.ops._C.cooperative_topk(
            logits, seq_lens, output, workspace, topk_tokens, max_seq_len
        )
        return

    if selected == "persistent":
        if not current_platform.is_cuda() or topk_tokens not in _NATIVE_TOPK_VALUES:
            raise ValueError(
                "persistent sparse-indexer top-k requires CUDA and "
                f"topk_tokens in {_NATIVE_TOPK_VALUES}"
            )
        (workspace,) = current_workspace_manager().get_simultaneous(
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )
        torch.ops._C.persistent_topk(
            logits, seq_lens, output, workspace, topk_tokens, max_seq_len
        )
        return

    if selected != "per_row":
        raise ValueError(f"unsupported sparse-indexer top-k backend: {selected}")
    torch.ops._C.top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        output,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )
