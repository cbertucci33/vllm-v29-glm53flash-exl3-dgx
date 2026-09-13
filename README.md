# GLM-5.3 Flash EXL3 on DGX Spark

This repository records the work required to serve a rank-sliced GLM-5.3 Flash EXL3 checkpoint with DFlash2 speculative decoding on two NVIDIA DGX Spark systems. It is based on vLLM 0.29.0. Model weights are published separately.

The runner supports GLM chat, reasoning, tools, multimodal input, hybrid KDA, native sparse MLA on GB10, EXL3 tensor parallelism, and DFlash2. It does not hard-code a context length, concurrency limit, KV-cache allocation, network address, or model path.

## How the runner was built

This is the integration history from unmodified vLLM to the tested runner. The order matters because later fixes depend on cache layouts and native interfaces established earlier.

1. **Started from vLLM 0.29.0.** The source base is tag `v0.29.0`, commit `98dff2a81d747d1dba01a47f939f48c3526d4206`. The runtime starts from the official `vllm/vllm-openai:v0.29.0` image pinned by digest.

2. **Added the GLM-5.3 Flash architecture.** Imported the model work from vLLM PR #53906, including hybrid KDA and NoPE sparse MLA, multimodal processing, MTP, configuration classes, registry entries, and weight-loading rules.

3. **Added the GLM serving protocol.** Packaged the official chat template from Z.ai model revision `690b705278a3a58e538fcb37c2ca8b5f9511213c`, retaining compatible reasoning, tool-call, parallel tool-result, and request-level thinking behavior.

4. **Replaced prebuilt FlashInfer with the required source build.** Built FlashInfer 0.6.18 from commit `5cc867a9bb560bc89b91dbc738a9f63f09beb89b`, which contains the native SM120/SM121 `GLM53_NOPE` sparse-MLA work from FlashInfer PRs #4802 and #4947.

5. **Matched vLLM to the native FlashInfer ABI.** Added the `FLASHINFER_MLA_SPARSE_SM120` backend and corrected packed FP8 cache records, physical sparse page tables, K/V scales, sequence lengths, indexer state, and decode metadata for GLM's NoPE layout.

6. **Corrected the target-cache format.** Selected `fp8_ds_mla` for native sparse MLA and used its real packed record size. Sliding-window layers can use their own cache format instead of inheriting the target MLA format.

7. **Added a GB10-safe exact TopK implementation.** The cooperative path exceeded the shared-memory limit on devices with less than 128 KiB per block. Added an exact non-cooperative streaming-radix path, including tie and threshold handling. Existing paths remain available on other GPUs.

8. **Added EXL3 to vLLM.** Implemented EXL3 configuration, tensor loading, prefill planning, logits handling, and routed expert integration.

9. **Added lossless tensor-parallel checkpoint slicing.** `tools/slice_exl3_checkpoint.py` splits routed expert tensors across ranks without dequantization or requantization. It reconstructs every source tensor bit for bit before publishing the output.

10. **Connected rank-sliced EXL3 to Sparkinfer.** Pinned Sparkinfer at commit `d4438d490691f79022fdfc8149e1c5f161d15445` and used its Trellis planning, scratch, binding, and execution interfaces for supported tensor-parallel shapes.

11. **Built ExLlamaV3 for ARM64.** Pinned commit `c5d9c657966ffeeaa9353f0cc899f18629da4a13`. Removed optional x86 AVX translation units from the ARM64 build, added fail-closed stubs for the unavailable CPU all-reduce path, and packaged only the extension consumed by vLLM.

12. **Updated CUTLASS DSL compatibility.** FlashInfer requires NVIDIA CUTLASS DSL 4.7.0. Sparkinfer's exact 4.6.0 dependency and B12X's exact 4.6.2 dependency were changed to `>=4.7.0,<5`. The final build pins 4.7.0.

13. **Added B12X dense MXFP8 dispatch.** Pinned B12X 1.2.6 at commit `ab6eea89b5b5e334ac6e9f2c503c1de60c3f216c`. B12X handles supported small batches, while FlashInfer Cutlass handles other shapes. Fixed the B12X capability check so `(False, reason)` does not evaluate as supported.

14. **Added DFlash2 for GLM.** Imported auxiliary target-state capture, compact MTP prefill, grouped convolutions, candidate selection, and the speculative decoding flow. The draft checkpoint uses ModelOpt MXFP8 weights and a rowwise FP8 draft head.

15. **Separated target and draft cache formats.** The target cache uses packed `fp8_ds_mla`; the DFlash GQA history ring uses `fp8_e4m3`. Draft pages no longer inherit target-page padding.

16. **Repaired the FlashKDA call boundary.** The pinned API requires caller-allocated output, final-state, and workspace buffers. The GLM prefill path now supplies all three. A focused test binds the caller to the pinned signature.

17. **Corrected DFlash page geometry.** Used the draft tensor's actual 16-token page size instead of the larger target sparse-MLA page size.

18. **Moved DFlash history into a private fixed ring.** DFlash history now uses bounded request-local storage and does not allocate shared target-pool blocks. Allocation, copy, zeroing, retirement, prefix resume, and worker page-table generation use the same ownership rule.

19. **Fixed scheduler admission for the private ring.** The scheduler no longer charges shared-pool block IDs for fixed DFlash storage. This removed an admission ceiling that could leave a continuation waiting while physical KV usage remained near zero.

20. **Fixed reported cache capacity.** Startup and metrics now exclude the private DFlash ring from shared-pool demand, matching the managers used for real request admission.

21. **Made native packaging source-aware.** FlashInfer JIT and packaged native artifacts are keyed to source identity and ABI inputs. An older cache entry can no longer hide a source or kernel change.

22. **Added complete warmup coverage.** Warmup covers the selected FlashInfer, EXL3, Sparkinfer, B12X, FlashKDA, TopK, and speculative rejection paths before CUDA graph capture.

23. **Allowed shorter DFlash2 inference blocks.** A checkpoint may run fewer proposals than its trained maximum, but it may not exceed that maximum. The tested checkpoint was trained with block size 8 (seven proposals) and was also smoke-tested with block size 6 (five proposals).

The static interface review is in [`docs/static-contract-review.md`](docs/static-contract-review.md). Source credits and licenses are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Pinned build inputs

`build/versions.env` is the machine-readable source of truth.

| Component | Pinned input |
| --- | --- |
| vLLM source | `0.29.0`, commit `98dff2a81d747d1dba01a47f939f48c3526d4206` |
| vLLM runtime image | `vllm/vllm-openai:v0.29.0@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` |
| CUDA build image | `nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6` |
| vLLM CUTLASS source | `da5e086dab31d63815acafdac9a9c5893b1c69e2` |
| FlashInfer | `0.6.18`, commit `5cc867a9bb560bc89b91dbc738a9f63f09beb89b` |
| FlashInfer CCCL | `16bd510c9b712e82b0ab6cbb630d8e29ba1f7116` |
| FlashInfer CUTLASS | `b46b16d003484063bca4ed365e44095c4c6ed633` |
| FlashInfer spdlog | `c3aed4b68373955e1cc94307683d44dca1515d2b` |
| Sparkinfer | `d4438d490691f79022fdfc8149e1c5f161d15445` |
| ExLlamaV3 | `c5d9c657966ffeeaa9353f0cc899f18629da4a13`, format `0.0.43` |
| B12X | `1.2.6`, commit `ab6eea89b5b5e334ac6e9f2c503c1de60c3f216c` |
| NVIDIA CUTLASS DSL | `4.7.0` |
| GLM chat template | Z.ai revision `690b705278a3a58e538fcb37c2ca8b5f9511213c` |

External sources and wheels are downloaded during build preparation. They are not vendored here.

## Prepare a tensor-parallel checkpoint

Inspect the split plan:

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

The output path must not exist. The tool writes to a temporary sibling, validates every tensor, then publishes the completed directory with one rename. It rejects incomplete expert records, non-MCG codebooks, unsupported bitrates, non-contiguous layers or experts, and dimensions that do not divide by the tensor-parallel size.

The qualified checkpoint used a two-way tensor-parallel split, 4-bit MCG EXL3, 43 MoE layers, 288 experts per layer, and 92 output shards. These values describe the tested artifact, not runner limits.

## Build the image

The final image starts from the pinned official vLLM runtime. A separate CUDA development stage compiles the changed vLLM extension and pinned external components. The build overlays the Python source and required native artifacts onto a fresh runtime stage.

See [`build/README.md`](build/README.md) for the reproducible build flow. The build downloads and hashes its inputs before offline compilation, then records source, wheel, extension, and toolchain provenance in the image.

## Serving configuration

The tested path uses these model-facing options:

```text
--quantization exl3
--load-format instanttensor
--attention-backend FLASHINFER_MLA_SPARSE_SM120
--linear-backend b12x
--kv-cache-dtype fp8_ds_mla
--kv-cache-dtype-skip-layers sliding_window
--block-size 2304
--prefix-match-unit 512
--speculative-config {"method":"dflash","model":"/path/to/dflash2","num_speculative_tokens":5}
--tool-call-parser glm47
--reasoning-parser glm45
--enable-auto-tool-choice
```

Five proposals were smoke-tested with the listed drafter. Seven proposals are the checkpoint's trained maximum. Select the proposal count from measured acceptance and end-to-end throughput for the intended workload.

Set model length, sequence concurrency, batch-token limits, KV memory, network addresses, ports, model paths, and chat defaults for the target deployment. Effective context and concurrency depend on checkpoint geometry, cache allocation, request mix, and available memory.

## Early measurements

These measurements come from one two-node DGX Spark deployment. They are not hardware limits or broad benchmark claims.

The longer real-use sample was collected with seven proposals:

| Measurement | Result |
| --- | --- |
| Completed requests | 197 of 197, all HTTP 200 |
| Prompt tokens | 24.35 million |
| Generated tokens | 283,800 |
| Weighted decode throughput | 24.5 tokens/s |
| Previous runtime on the same hardware | 21.62 tokens/s |
| Overall change | +13.3% |
| Matched 40K to 100K prompts | 26.02 vs 22.65 tokens/s, +14.9% |
| Matched 100K to 200K prompts | 24.22 vs 21.55 tokens/s, +12.4% |
| Prefix-cache reuse | 93.6% |
| Errors, disconnects, or capacity waits | 0 |

The five-proposal update has a bounded smoke test, not a long-run performance claim:

| Measurement | Result |
| --- | --- |
| Draft accounting | 263 verification steps x 5 proposals = 1,315 drafts |
| Draft positions exposed | positions 0 through 4 only |
| Text completion | HTTP 200, normal stop, correct visible answer |
| Code completion | HTTP 200, normal stop, correct expression |
| Tool use | HTTP 200, valid structured tool call |
| Runtime state | no waits, restarts, OOMs, or logged errors |

## Validation

Completed checks include:

- static caller and dependency interface review for the selected path;
- focused CPU and CUDA tests for changed cache and kernel boundaries;
- exact GB10 TopK tests, including ties and threshold cases;
- lossless rank-sliced checkpoint reconstruction checks;
- full two-rank checkpoint load;
- FlashInfer, Sparkinfer, EXL3, FlashKDA, B12X, warmup, and CUDA graph capture;
- OpenAI-compatible chat, code, and structured tool-call requests;
- multi-turn tool-use smoke testing;
- continuation beyond the original scheduler-admission failure;
- private-ring capacity reporting through the real startup path;
- five-proposal DFlash2 startup, graph capture, and exact draft accounting.

These checks do not establish every workload, context length, concurrency level, multimodal shape, or hardware revision.

## License

The vLLM-derived source remains under the Apache License 2.0 in `LICENSE`. External dependencies and model artifacts retain their own licenses and terms.
