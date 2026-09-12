# SPDX-License-Identifier: Apache-2.0
"""Aux-hidden capture for glm5_next DFlash drafting.

Capture contract (sglang PR #36708, the draft authors' serving
implementation): the completed output of target layer k with the mHC streams
mean-contracted to one per-token state. The fork's deferred-mHC dialect never
materializes that tensor between layers, so Glm5NextModel composes it at the
capture boundary; these tests pin the composition to the reference
formulation from the HF implementation (Glm5NextTextDecoderLayer's
post/comb combine + Glm5NextTextHyperHead's unweighted stream mean).
"""

from types import SimpleNamespace

import torch

import vllm.model_executor.kernels.mhc as mhc_kernels
from vllm.models.glm5next.nvidia.model import Glm5NextModel
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator


def _reference_completed_output(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    full = post * x.unsqueeze(-2) + comb.transpose(-1, -2) @ residual
    return full.mean(dim=-2)


def test_completed_layer_output_matches_reference_formula():
    torch.manual_seed(0)
    s, n, h = 5, 4, 16
    x = torch.randn(s, h)
    residual = torch.randn(s, n, h)
    post = torch.rand(s, n, 1)
    comb = torch.rand(s, n, n)

    layer = SimpleNamespace(
        n=n,
        hc_post=lambda x, r, p, c: mhc_kernels.mhc_post_torch(x, r, p, c),
    )
    actual = Glm5NextModel._completed_layer_output(None, layer, x, residual, post, comb)
    torch.testing.assert_close(
        actual,
        _reference_completed_output(x, residual, post, comb),
        rtol=1e-5,
        atol=1e-5,
    )


def test_completed_layer_output_passthrough_without_mhc_state():
    x = torch.randn(3, 8)
    out = Glm5NextModel._completed_layer_output(
        None, SimpleNamespace(), x, None, None, None
    )
    assert out is x


def test_glm5next_exposes_eagle3_capture_interface():
    from vllm.model_executor.models.interfaces import EagleModelMixin
    from vllm.models.glm5next.nvidia.model import (
        Glm5NextForCausalLM,
        Glm5NextForConditionalGeneration,
    )

    assert issubclass(Glm5NextModel, EagleModelMixin)
    assert getattr(Glm5NextForCausalLM, "supports_eagle3", False)
    assert getattr(Glm5NextForConditionalGeneration, "supports_eagle3", False)


def test_mtp_compact_prefill_indices_are_scoped_to_prefill():
    class Recorder:
        def __init__(self):
            self.calls = []

        def set_prefill_output_indices(self, indices):
            self.calls.append(None if indices is None else indices.clone())

    recorder = Recorder()
    speculator = object.__new__(MTPSpeculator)
    speculator.model = SimpleNamespace(model=recorder)
    speculator.prefill_outputs_are_compact = True
    speculator.share_mtp_topk_indices = False
    speculator.last_token_indices = torch.tensor([4, 9, 12], dtype=torch.int64)

    speculator.on_prefill_begin(2)
    speculator.on_prefill_end(2)

    assert len(recorder.calls) == 2
    torch.testing.assert_close(recorder.calls[0], torch.tensor([4, 9]))
    assert recorder.calls[1] is None
