# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config-only DFlash behavior.

``dflash_has_any_non_causal`` decides pre-build whether the draft needs a
non-causal-capable backend, so its branch table (explicit override, SWA-derived
per-layer causality, and the no-``layer_types`` fallback) is worth pinning.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.qwen3_dflash import (
    DFlashAttention,
    _dflash_layer_causal,
    _get_dflash_fc_input_size,
    dflash_has_any_non_causal,
)
from vllm.v1.kv_cache_interface import DFlashSWASpec, SlidingWindowSpec
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)


def _config(
    num_hidden_layers,
    layer_types=None,
    causal_override=None,
    is_causal=None,
    architectures=None,
):
    dflash_config = None if causal_override is None else {"causal": causal_override}
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        dflash_config=dflash_config,
        is_causal=is_causal,
        architectures=architectures,
    )


@pytest.mark.parametrize(
    "config,expected",
    [
        # Override forces causality on every layer, ignoring layer_types.
        (_config(2, layer_types=["full_attention"] * 2, causal_override=True), False),
        # Override forces non-causal on every layer.
        (
            _config(2, layer_types=["sliding_attention"] * 2, causal_override=False),
            True,
        ),
        # DFlash2 stores the explicit attention semantics at the top level.
        (
            _config(
                2,
                layer_types=["sliding_attention"] * 2,
                is_causal=False,
                architectures=["DFlash2DraftModel"],
            ),
            True,
        ),
        (
            _config(
                2,
                layer_types=["full_attention"] * 2,
                is_causal=True,
                architectures=["DFlash2DraftModel"],
            ),
            False,
        ),
        # A v1 draft config carrying a stray is_causal (this fork overloads
        # that attr for pooling models) must keep its layer_types derivation.
        (
            _config(2, layer_types=["sliding_attention"] * 2, is_causal=False),
            False,
        ),
        # SWA-derived: full-attention layers are non-causal.
        (_config(2, layer_types=["sliding_attention", "full_attention"]), True),
        # SWA-derived: all-sliding is fully causal.
        (_config(2, layer_types=["sliding_attention", "sliding_attention"]), False),
        # No layer_types -> non-causal fallback.
        (_config(2, layer_types=None), True),
        (_config(2, layer_types=[]), True),
    ],
)
def test_dflash_has_any_non_causal(config, expected):
    assert dflash_has_any_non_causal(config) is expected


def test_dflash_spec_drops_target_page_padding(monkeypatch):
    padded = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        sliding_window=32_768,
        page_size_padded=4_718_592,
    )
    monkeypatch.setattr(Attention, "get_kv_cache_spec", lambda *_args: padded)

    attention = object.__new__(DFlashAttention)
    spec = attention.get_kv_cache_spec(SimpleNamespace())

    assert isinstance(spec, DFlashSWASpec)
    assert spec.block_size == 16
    assert spec.page_size_padded is None
    assert spec.page_size_bytes == 16_384


def test_dflash_layer_causal_is_per_layer():
    config = _config(2, layer_types=["sliding_attention", "full_attention"])
    assert _dflash_layer_causal(config, 0) is True
    assert _dflash_layer_causal(config, 1) is False


def test_dflash_layer_causal_honors_top_level_override():
    config = _config(
        2,
        layer_types=["sliding_attention", "full_attention"],
        is_causal=False,
        architectures=["DFlash2DraftModel"],
    )
    assert _dflash_layer_causal(config, 0) is False
    assert _dflash_layer_causal(config, 1) is False


def test_dflash_layer_causal_ignores_is_causal_on_v1_drafts():
    config = _config(
        2,
        layer_types=["sliding_attention", "full_attention"],
        is_causal=True,
    )
    assert _dflash_layer_causal(config, 0) is True
    assert _dflash_layer_causal(config, 1) is False


def _vllm_config(**draft_config):
    config = SimpleNamespace(**draft_config)
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        )
    )


def test_dflash_fc_uses_aux_layer_count():
    vllm_config = _vllm_config(
        num_hidden_layers=5,
        hidden_size=4096,
        target_hidden_size=None,
        target_layer_ids=[1, 17, 32],
    )

    assert _get_dflash_fc_input_size(vllm_config) == 3 * 4096


@pytest.mark.parametrize("config_name", ["dflash_config", "eagle_config"])
def test_eagle_aux_layers_preserves_legacy_layer_ids(config_name):
    layer_ids = [1, 17, 32]
    vllm_config = _vllm_config(
        **{config_name: {"layer_ids": layer_ids}},
    )

    assert get_eagle3_aux_layers_from_config(vllm_config.speculative_config) == tuple(
        layer_ids
    )
