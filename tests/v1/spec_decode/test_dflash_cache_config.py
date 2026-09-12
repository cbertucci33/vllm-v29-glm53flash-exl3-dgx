# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.config import AttentionConfig, CacheConfig
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash.utils import get_dflash_cache_config


def _vllm_config(
    target_dtype: str,
    draft_dtype: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=CacheConfig(cache_dtype=target_dtype),
        speculative_config=SimpleNamespace(kv_cache_dtype=draft_dtype),
    )


def test_packed_mla_target_uses_standard_fp8_dflash_cache() -> None:
    config = _vllm_config("fp8_ds_mla")

    draft_cache = get_dflash_cache_config(config)

    assert config.cache_config.cache_dtype == "fp8_ds_mla"
    assert draft_cache.cache_dtype == "fp8_e4m3"
    assert draft_cache is not config.cache_config


def test_explicit_dflash_cache_dtype_wins() -> None:
    config = _vllm_config("fp8_ds_mla", "bfloat16")

    draft_cache = get_dflash_cache_config(config)

    assert draft_cache.cache_dtype == "bfloat16"
    assert config.cache_config.cache_dtype == "fp8_ds_mla"


def test_compatible_target_dtype_is_copied_without_aliasing() -> None:
    config = _vllm_config("auto")

    draft_cache = get_dflash_cache_config(config)

    assert draft_cache.cache_dtype == "auto"
    assert draft_cache is not config.cache_config


def test_dflash_metadata_builders_receive_draft_cache_config() -> None:
    config = _vllm_config("fp8_ds_mla")
    config.attention_config = AttentionConfig()
    speculator = object.__new__(DFlashSpeculator)
    speculator.vllm_config = config
    speculator.requires_non_causal = True

    draft_config = speculator.attn_vllm_config

    assert draft_config.cache_config.cache_dtype == "fp8_e4m3"
    assert draft_config.cache_config is not config.cache_config
    assert draft_config.attention_config.use_non_causal is True
    assert config.attention_config.use_non_causal is False
