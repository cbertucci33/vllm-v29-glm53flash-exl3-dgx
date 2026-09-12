# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coverage for the dense MXFP8 B12X row-count dispatch boundary."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp8 import b12x as b12x_mod
from vllm.model_executor.kernels.linear.mxfp8.b12x import B12xMxfp8LinearKernel


@pytest.fixture
def fake_layer() -> SimpleNamespace:
    return SimpleNamespace(
        b12x_mxfp8_packed_weight=SimpleNamespace(out_features=16),
        b12x_mxfp8_fallback_weight=torch.zeros(16, 32),
        b12x_mxfp8_fallback_scale=torch.zeros(16),
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, int]]
) -> None:
    def b12x_mm(source, packed_weight, **kwargs):
        del kwargs
        calls.append(("b12x", int(source.shape[0])))
        return torch.zeros(source.shape[0], packed_weight.out_features)

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_mxfp8",
        lambda: SimpleNamespace(mm=b12x_mm),
    )
    monkeypatch.setattr(
        b12x_mod,
        "_apply_flashinfer_mxfp8_fallback",
        lambda source, bias, weight, scale: (
            calls.append(("flashinfer", int(source.shape[0])))
            or torch.zeros(source.shape[0], weight.shape[0])
        ),
    )


@pytest.mark.parametrize(
    ("rows", "expected_backend"),
    [(7, "b12x"), (14, "b12x"), (16, "b12x"), (17, "flashinfer")],
)
def test_mxfp8_dispatch_uses_configured_row_cutoff(
    monkeypatch: pytest.MonkeyPatch,
    fake_layer: SimpleNamespace,
    rows: int,
    expected_backend: str,
) -> None:
    monkeypatch.setattr(b12x_mod.envs, "VLLM_B12X_MXFP8_MAX_M", 16)
    calls: list[tuple[str, int]] = []
    _install_fakes(monkeypatch, calls)

    output = b12x_mod._apply_b12x_mxfp8_packed_linear(
        fake_layer, torch.zeros(rows, 32), None
    )

    assert output.shape == (rows, 16)
    assert calls == [(expected_backend, rows)]


def test_mxfp8_dispatch_zero_cutoff_keeps_b12x(
    monkeypatch: pytest.MonkeyPatch,
    fake_layer: SimpleNamespace,
) -> None:
    monkeypatch.setattr(b12x_mod.envs, "VLLM_B12X_MXFP8_MAX_M", 0)
    calls: list[tuple[str, int]] = []
    _install_fakes(monkeypatch, calls)

    b12x_mod._apply_b12x_mxfp8_packed_linear(
        fake_layer, torch.zeros(64, 32), None
    )

    assert calls == [("b12x", 64)]


def test_weight_processing_retains_large_row_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed = SimpleNamespace(out_features=128)
    monkeypatch.setattr(b12x_mod.envs, "VLLM_B12X_MXFP8_MAX_M", 16)
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_mxfp8",
        lambda: SimpleNamespace(pack_weight=lambda weight, scale: packed),
    )
    import vllm.model_executor.layers.quantization.utils.mxfp8_utils as mxfp8_utils
    import vllm.utils.flashinfer as vllm_flashinfer

    monkeypatch.setattr(vllm_flashinfer, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        mxfp8_utils,
        "swizzle_mxfp8_scale",
        lambda scale, **kwargs: scale.clone(),
    )

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((128, 128), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.zeros((128, 4), dtype=torch.uint8),
        requires_grad=False,
    )
    kernel = object.__new__(B12xMxfp8LinearKernel)
    kernel.process_weights_after_loading(layer)

    assert layer.b12x_mxfp8_packed_weight is packed
    assert layer.b12x_mxfp8_fallback_weight.shape == (128, 128)
    assert layer.b12x_mxfp8_fallback_scale.shape == (128, 4)
    assert layer.weight.numel() == 0
    assert layer.weight_scale.numel() == 0

    calls: list[tuple[str, int]] = []
    _install_fakes(monkeypatch, calls)
    output = kernel.apply_weights(layer, torch.zeros(17, 128))

    assert output.shape == (17, 128)
    assert calls == [("flashinfer", 17)]
