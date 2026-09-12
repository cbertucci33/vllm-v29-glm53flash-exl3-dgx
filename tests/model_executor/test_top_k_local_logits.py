# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity coverage for the FP8 draft head's preprojected logits path."""

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)


def _fake_lm_head(weight: torch.Tensor, vocab: int, padded_vocab: int):
    shard = VocabParallelEmbeddingShardIndices(
        padded_org_vocab_start_index=0,
        padded_org_vocab_end_index=padded_vocab,
        padded_added_vocab_start_index=padded_vocab,
        padded_added_vocab_end_index=padded_vocab,
        org_vocab_start_index=0,
        org_vocab_end_index=vocab,
        added_vocab_start_index=vocab,
        added_vocab_end_index=vocab,
    )
    quant_method = SimpleNamespace(
        apply=lambda layer, x, bias=None: torch.nn.functional.linear(x, layer.weight)
    )
    return SimpleNamespace(
        weight=weight, tp_size=1, shard_indices=shard, quant_method=quant_method
    )


def test_local_logits_match_head_projection():
    torch.manual_seed(0)
    vocab, padded_vocab, hidden, k = 100, 128, 32, 8
    weight = torch.randn(padded_vocab, hidden)
    lm_head = _fake_lm_head(weight, vocab, padded_vocab)
    x = torch.randn(5, hidden)
    fake_self = SimpleNamespace(
        scale=1.0,
        soft_cap=None,
        head_dtype=None,
        _apply_head=lambda lm_head, hs, bias: LogitsProcessor._apply_head(
            fake_self, lm_head, hs, bias
        ),
    )

    ids_ref, vals_ref = LogitsProcessor.get_top_k_tokens(fake_self, lm_head, x, k)

    projected = torch.nn.functional.linear(x, weight)
    calls = []
    fake_self._apply_head = lambda *args, **kwargs: calls.append(1)
    ids, vals = LogitsProcessor.get_top_k_tokens(
        fake_self, lm_head, x, k, local_logits=projected.clone()
    )

    assert not calls, "local_logits must bypass the head projection"
    torch.testing.assert_close(ids, ids_ref)
    torch.testing.assert_close(vals, vals_ref)
    assert int(ids.max()) < vocab, "padding columns must stay masked"
