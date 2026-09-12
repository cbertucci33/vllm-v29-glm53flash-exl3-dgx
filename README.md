# GLM-5.3 Flash EXL3 on DGX Spark

This repository extends vLLM 0.29.0 to serve GLM-5.3 Flash EXL3 checkpoints
on NVIDIA DGX Spark. It includes native SM120 sparse attention, tensor-parallel
EXL3 checkpoint slicing and loading, DFlash2 speculative decoding, and the
cache lifecycle changes required by that combination.

Model weights are published separately. The source checkpoint and the DFlash2
checkpoint remain subject to their own licenses.

## Component versions

| Component | Version or revision | Use |
| --- | --- | --- |
| vLLM | `0.29.0`, commit `98dff2a81d747d1dba01a47f939f48c3526d4206` | Runtime base |
| Official vLLM image | `vllm/vllm-openai:v0.29.0@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` | Runtime stage |
| CUDA development image | `nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6` | Build toolchain |
| CUTLASS C++ source | `da5e086dab31d63815acafdac9a9c5893b1c69e2` | vLLM CUDA extension |
| FlashInfer | `0.6.18`, commit `5cc867a9bb560bc89b91dbc738a9f63f09beb89b` | SM120 sparse attention and TopK |
| FlashInfer CCCL | `16bd510c9b712e82b0ab6cbb630d8e29ba1f7116` | Pinned source dependency |
| FlashInfer CUTLASS | `b46b16d003484063bca4ed365e44095c4c6ed633` | Pinned source dependency |
| FlashInfer spdlog | `c3aed4b68373955e1cc94307683d44dca1515d2b` | Pinned source dependency |
| Sparkinfer | commit `d4438d490691f79022fdfc8149e1c5f161d15445` | Tensor-parallel EXL3 execution |
| ExLlamaV3 | commit `c5d9c657966ffeeaa9353f0cc899f18629da4a13` | EXL3 extension, format `0.0.43` |
| B12X | `1.2.6`, commit `ab6eea89b5b5e334ac6e9f2c503c1de60c3f216c` | Dense MXFP8 linear kernels |
| NVIDIA CUTLASS DSL | `4.7.0` | Kernel build dependency |
| GLM chat template | Z.ai revision `690b705278a3a58e538fcb37c2ca8b5f9511213c` | Chat, tool, vision, and reasoning syntax |

`build/versions.env` contains the machine-readable pins. External sources and
build wheels are downloaded separately; they are not vendored in this
repository.

## Changes from vLLM 0.29.0

1. **GLM-5.3 model support.** The model implementation includes hybrid KDA,
   NoPE sparse MLA, multimodal input, MTP, and the GLM parser and configuration
   surface.
2. **Native SM120 attention.** The target uses FlashInfer's `GLM53_NOPE`
   backend with packed FP8 cache records and physical sparse page tables.
3. **Lossless tensor-parallel EXL3 slicing.** The included converter splits
   routed expert tensors without dequantization, calibration, or
   requantization. It reconstructs every source tensor bit for bit before
   publishing the output.
4. **Tensor-parallel EXL3 loading.** The loader normalizes target and draft
   tensor names and dispatches supported rank-sliced shapes through
   Sparkinfer's Trellis interface.
5. **DFlash2 integration.** The runtime captures GLM auxiliary state, loads the
   ModelOpt MXFP8 draft, runs compact MTP and FlashKDA prefill, and uses a
   rowwise FP8 draft head.
6. **Private draft KV ring.** DFlash history uses bounded request-local storage
   instead of shared target-cache blocks. Scheduler allocation, request
   retirement, prefix resume, block copy, zeroing, and page-table generation
   follow the same ownership rule.
7. **Correct shared-cache accounting.** Fixed DFlash storage is excluded from
   shared-pool capacity calculations.
8. **Dense MXFP8 dispatch.** B12X handles supported small batches;
   FlashInfer Cutlass handles other shapes.
9. **FlashKDA API repair.** The GLM caller supplies the allocated output,
   final-state, and workspace buffers required by the pinned API.
10. **Cache geometry repairs.** The implementation uses the packed target-cache
    record size, sparse page-table size, and DFlash page size required by the
    real tensors.
11. **GB10 TopK.** Devices with less than 128 KiB shared memory per block use
    an exact non-cooperative streaming-radix path. Existing paths remain in
    place for other GPUs.
12. **Source-keyed native packaging.** Native FlashInfer artifacts are tied to
    source identity so an older cache entry cannot hide a source or ABI change.

The static interface review is in
[`docs/static-contract-review.md`](docs/static-contract-review.md). Dependency
sources and licenses are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Prepare a tensor-parallel checkpoint

Inspect the planned split:

```bash
python3 tools/slice_exl3_checkpoint.py \
  /path/to/source-checkpoint \
  --tp 2 \
  --plan
```

Create the checkpoint:

```bash
python3 tools/slice_exl3_checkpoint.py \
  /path/to/source-checkpoint \
  /path/to/rank-sliced-checkpoint \
  --tp 2
```

The output path must not exist. The tool writes to a temporary sibling,
validates every tensor, then publishes the completed directory with one
rename. It rejects incomplete expert records, non-MCG codebooks, unsupported
bitrates, non-contiguous layers or experts, and dimensions that do not divide
by the tensor-parallel size.

The qualified checkpoint used a two-way tensor-parallel split, 4-bit MCG EXL3,
43 MoE layers, 288 experts per layer, and 92 output shards. These describe the
tested model artifact, not fixed runner limits.

## Build the image

The build starts from the pinned official vLLM image. It compiles the changed
vLLM CUDA extension and pinned external components, then overlays the Python
source and native artifacts onto a fresh runtime stage. Unchanged native vLLM
artifacts remain from the official image.

See [`build/README.md`](build/README.md) for the reproducible build flow. Build
inputs are pinned by digest or commit, downloaded before compilation, hashed,
and compiled offline. The final image records source, wheel, extension, and
toolchain provenance.

## Serving configuration

The qualified path uses these model-facing options:

```text
--quantization exl3
--load-format instanttensor
--attention-backend FLASHINFER_MLA_SPARSE_SM120
--linear-backend b12x
--kv-cache-dtype fp8_ds_mla
--kv-cache-dtype-skip-layers sliding_window
--block-size 2304
--prefix-match-unit 512
--speculative-config {"method":"dflash","model":"/path/to/dflash2","num_speculative_tokens":7}
--tool-call-parser glm47
--reasoning-parser glm45
--enable-auto-tool-choice
```

Set model length, sequence concurrency, batch-token limits, KV memory, network
addresses, ports, model paths, and chat defaults for the target deployment.
They are not hard-coded capabilities of this runner. Effective context and
concurrency depend on checkpoint geometry, cache allocation, request mix, and
available memory.

The target cache uses packed `fp8_ds_mla`. The DFlash GQA ring uses
`fp8_e4m3`; it does not inherit the target-cache format.

## Early measurements

These results come from one two-node DGX Spark tensor-parallel deployment.
They are initial measurements, not hardware limits or broad benchmark claims.

| Measurement | Result |
| --- | --- |
| Multi-turn tool-use smoke test | 8 of 8 checks passed |
| Smoke-test activity | 15 model turns and 17 tool calls |
| Smoke-test token volume | 335,046 prompt tokens and 18,672 completion tokens |
| Smoke-test wall time | 521.94 seconds |
| Request-wall completion throughput | 36.49 completion tokens/s |
| Baseline on the same two-node hardware | 32.25 completion tokens/s |
| Change from baseline | +13.2% |
| DFlash acceptance | 14,398 of 29,911 drafted tokens, 48.1% |
| Accepted draft tokens per verification step | 3.37 |
| Prefix-cache reuse during the smoke test | 258,048 of 335,046 prompt tokens, 77.0% |

The throughput comparison is directional because the runs produced different
output mixes. The DFlash acceptance result applies to the tested checkpoint
and drafter pairing.

## Validation

Completed checks include:

- static caller and dependency interface review for the selected path;
- focused CPU and CUDA tests for changed cache and kernel boundaries;
- exact GB10 TopK tests, including ties and threshold cases;
- full two-rank checkpoint load;
- FlashInfer, Sparkinfer, EXL3, FlashKDA, B12X, warmup, and CUDA graph capture;
- OpenAI-compatible API readiness and a bounded chat canary;
- a multi-turn tool-use smoke test;
- continuation beyond the original scheduler-admission failure;
- corrected private-ring capacity reporting through the startup path.

These checks do not establish every workload, context length, concurrency
level, multimodal shape, or hardware revision.

## License

The vLLM-derived source remains under the Apache License 2.0 in `LICENSE`.
External dependencies and model artifacts retain their own licenses and terms.
