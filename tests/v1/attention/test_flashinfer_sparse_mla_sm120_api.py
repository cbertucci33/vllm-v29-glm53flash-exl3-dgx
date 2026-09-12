# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    _required_sm120_sparse_topk,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm120 as sm120
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(
    model_type: str = "glm5_next_text",
    **geometry,
) -> SimpleNamespace:
    fields = {
        "model_type": model_type,
        "index_topk": 2048,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "kv_lora_rank": 512,
    }
    fields.update(geometry)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(**fields),
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_sm120_native_backend_precedes_generic_triton_on_sm121() -> None:
    cuda_platform = pytest.importorskip("vllm.platforms.cuda", exc_type=ImportError)
    priorities = cuda_platform._get_backend_priorities(
        use_mla=True,
        device_capability=DeviceCapability(12, 1),
        kv_cache_dtype="fp8_ds_mla",
        head_size=512,
    )
    assert priorities[:2] == [
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
        AttentionBackendEnum.TRITON_MLA,
    ]


def test_sm120_backend_uses_sparse_mqa_for_prefill() -> None:
    impl_cls = FlashInferMLASparseSM120Backend.get_impl_cls()

    assert impl_cls.is_sparse
    assert not impl_cls.supports_dense_mha_prefill


def test_glm_sm120_backend_accepts_packed_glm53_nope_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config()):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


@pytest.mark.parametrize("kv_cache_dtype", [None, "auto", "fp8", "fp8_e4m3"])
def test_glm_sm120_backend_rejects_nonpacked_cache(
    monkeypatch, kv_cache_dtype
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    with set_current_vllm_config(_fake_vllm_config()):
        reason = FlashInferMLASparseSM120Backend.supports_combination(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype=kv_cache_dtype,
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            device_capability=DeviceCapability(12, 0),
        )
    assert reason is not None
    assert "fp8_ds_mla" in reason


@pytest.mark.parametrize(
    ("head_size", "use_sparse"), [(576, True), (512, False)]
)
def test_glm_sm120_backend_rejects_wrong_attention_contract(
    monkeypatch, head_size, use_sparse
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    with set_current_vllm_config(_fake_vllm_config()):
        reason = FlashInferMLASparseSM120Backend.supports_combination(
            head_size=head_size,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=use_sparse,
            use_mm_prefix=False,
            device_capability=DeviceCapability(12, 1),
        )
    assert reason is not None


@pytest.mark.parametrize(
    ("model_type", "geometry"),
    [
        ("deepseek_v3", {}),
        ("glm5_next_text", {"index_topk": 1024}),
        ("glm5_next_text", {"qk_nope_head_dim": 512}),
        ("glm5_next_text", {"qk_rope_head_dim": 64}),
        ("glm5_next_text", {"kv_lora_rank": 256}),
    ],
)
def test_glm_sm120_backend_rejects_wrong_model_or_geometry(
    monkeypatch, model_type, geometry
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    with set_current_vllm_config(_fake_vllm_config(model_type, **geometry)):
        reason = FlashInferMLASparseSM120Backend.supports_combination(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            device_capability=DeviceCapability(12, 0),
        )
    assert reason is not None


def test_sm120_impl_accepts_glm_nope_packed_cache(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    topk_indices_buffer = torch.full((1, 2176), -1, dtype=torch.int32)

    with set_current_vllm_config(_fake_vllm_config("glm5_next_text")):
        impl = sm120.FlashInferMLASparseSM120Impl(
            num_heads=32,
            head_size=512,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_ds_mla",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            topk_indices_buffer=topk_indices_buffer,
            kv_lora_rank=512,
            qk_nope_head_dim=256,
            qk_rope_head_dim=0,
        )

    assert impl.qk_rope_head_dim == 0
    assert impl.kv_scale_format == "arbitrary_fp32"
    assert impl.topk_indices_buffer is topk_indices_buffer


def test_sm120_dsv4_capability_checks_exact_dispatch_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        _DECODE_DSV4_DISPATCH=frozenset({(32, 128), (32, 192)})
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 128)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 192)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 256)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(16, 192)

    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()


def test_sm120_dsv4_required_topk_tracks_dspark_width() -> None:
    causal = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    dspark = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )

    assert _required_sm120_sparse_topk(causal, 128) == 128
    assert _required_sm120_sparse_topk(dspark, 128) == 192


def test_sm120_forward_uses_physical_glm_topk_capacity(monkeypatch) -> None:
    impl = object.__new__(sm120.FlashInferMLASparseSM120Impl)
    impl.topk_indices_buffer = torch.full((1, 2176), -1, dtype=torch.int32)
    impl.num_heads = 1
    impl.kv_lora_rank = 512
    impl.qk_nope_head_dim = 256
    impl.qk_rope_head_dim = 0
    impl.scale = 1.0
    impl.kv_scale_format = "arbitrary_fp32"
    impl._workspace_buffer = None

    physical_topk = torch.full((1, 2176), -1, dtype=torch.int32)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        sm120,
        "triton_convert_req_index_to_global_index",
        lambda *_args, **_kwargs: physical_topk,
    )
    monkeypatch.setattr(
        sm120,
        "_get_workspace_buffer",
        lambda _device: torch.empty(0),
    )

    def fake_decode(**kwargs):
        captured.update(kwargs)
        return kwargs["out"]

    monkeypatch.setattr(
        fi_utils,
        "flashinfer_trtllm_batch_decode_with_kv_cache_mla",
        fake_decode,
    )

    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        block_size=64,
        topk_tokens=2048,
    )
    q = torch.zeros((1, 512), dtype=torch.bfloat16)
    kv_cache = torch.zeros((1, 1, 656), dtype=torch.uint8)

    output, _ = impl.forward_mqa(q, kv_cache, metadata, layer=None)  # type: ignore[arg-type]

    assert output.shape == (1, 1, 512)
    assert captured["max_seq_len"] == 2176
    assert captured["sparse_mla_top_k"] == 2176
    assert cast(torch.Tensor, captured["block_tables"]).shape == (1, 1, 2176)


def test_sm120_workspace_is_allocated_during_kernel_warmup(monkeypatch) -> None:
    from vllm.model_executor.warmup import kernel_warmup as warmup_module
    from vllm.v1.attention.backends.mla import flashinfer_mla_sparse

    device = torch.device("cpu")
    backend = SimpleNamespace(get_name=lambda: "FLASHINFER_MLA_SPARSE_SM120")
    group = SimpleNamespace(backend=backend)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(device=device, attn_groups=[[group]])
    )
    captured: list[torch.device] = []
    monkeypatch.setattr(
        flashinfer_mla_sparse,
        "_get_workspace_buffer",
        lambda actual_device: captured.append(actual_device),
    )

    warmup_module._warmup_flashinfer_sm120_sparse_mla_workspace(worker)

    assert captured == [device]
