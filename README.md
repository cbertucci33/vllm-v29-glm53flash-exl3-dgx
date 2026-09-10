# vLLM v0.29 for GLM-5.3 Flash on native SM120

This repository develops a GLM-5.3 Flash runner for two NVIDIA DGX Sparks. It
starts from vanilla vLLM 0.29 and uses FlashInfer's native SM120/SM121
`GLM53_NOPE` sparse-attention path.

No production deployment should be built from an unaudited commit.

## Baseline

- vLLM release: `v0.29.0`
- Base commit: `98dff2a81d747d1dba01a47f939f48c3526d4206`
- Target hardware: two NVIDIA DGX Sparks using tensor parallelism over
  ConnectX-7
- Planned target model: GLM-5.3 Flash with a rank-sliced EXL3 target and an
  MXFP8 DFlash2 drafter

## Current source state

The current source contains the GLM target-model baseline, native SM120
sparse-MLA prerequisites, and rank-sliced EXL3 execution:

- upstream GLM-5.3 Flash support from vLLM PR #53906, retained with Jiangyun
  Zhu's authorship;
- GLM NoPE packed-cache support from vLLM PR #55277 commit `0fa2b3308c`;
- physical 2,176-entry sparse page-table sizing from vLLM PR #55277 commit
  `8d09804c87`;
- K-pool sentinel initialization and bounds checks derived from Entrpi commit
  `5fc13966f5`, with unrelated SM90 compatibility changes excluded;
- rank-sliced EXL3 loading, target and MTP weight-name normalization, and
  planned Trellis execution through Sparkinfer PR #49 commit
  `d4438d490691f79022fdfc8149e1c5f161d15445`.

The FlashInfer and Sparkinfer commits below are documented build requirements.
This source does not yet package either dependency, DFlash2, a drafter, or a
launch configuration.

## Native SM120 integration

The target attention path will use:

- FlashInfer commit `5cc867a9bb560bc89b91dbc738a9f63f09beb89b`, which
  contains the native `GLM53_NOPE` kernel and graph-capture correction;
- the GLM NoPE and physical page-table changes from vLLM PR #55277;
- packed `fp8_ds_mla` target KV records;
- GLM's 2,176-entry physical sparse page-table capacity.

This replaces the SM90 compatibility route used by v1. It does not import the
old B12X sparse-MLA backend or Entrpi's expanded physical K-pool design.

## Required imported subsystems

The runner still needs work that is not part of vanilla vLLM 0.29:

- upstream GLM-5.3 model support;
- rank-sliced EXL3/Trellis execution through pinned Sparkinfer PR #49. Every
  rank-sliced batch shape uses its planned Trellis path. This route does not
  use the legacy ExLlamaV3 routed-expert fallback;
- GLM DFlash2 support and the fixed non-prefix-cacheable drafter KV ring;
- independently applicable GLM, SM121, cache, and scheduler corrections.

Imported commits will retain their original author metadata and source links.
`THIRD_PARTY_NOTICES.md` records the exact revisions and licenses.

## Work authored in this repository

The local integration is responsible for:

- adapting the native SM120 cache writer and planner to vLLM 0.29;
- keeping the packed target cache separate from the ordinary GQA DFlash ring;
- tracing cache allocation, registration, hashing, copying, zeroing, and graph
  capture across both layouts;
- selecting B12X for eligible small MXFP8 linear batches, with FlashInfer as
  the fallback for larger or unsupported shapes;
- removing obsolete flags, compatibility routes, imports, and dead code;
- pinning the image toolchain and checking source-to-image provenance;
- adding focused contract tests for every changed boundary.

## Excluded architecture

The production v2 path will not use:

- `FLASHINFER_MLA_SPARSE_SM90` for GLM target attention;
- Entrpi's old B12X sparse-MLA backend;
- the old expanded physical K-pool storage path;
- the obsolete `VLLM_USE_B12X_FP8_GEMM` flag;
- implicit fallbacks that hide a missing native dependency.

Generic upstream code may remain in the tree when other models use it. The GLM
v2 construction and launch paths must not select it.

## Acceptance boundary

Before a live test, the repository must pass:

1. A full source and codebase audit covering construction, backend selection,
   cache layout, K-pool geometry, DFlash, EXL3, graph capture, image packaging,
   and the two-rank launch contract.
2. Focused CPU and GPU contract tests for the changed interfaces.
3. Source-to-image hash and dependency read-back checks.

Live acceptance will use one exact-shape SM121 canary followed by one
representative two-rank Clara workload. The current v1 image remains the
rollback baseline.

## License

The repository follows the upstream vLLM Apache 2.0 license in `LICENSE`.
Separate dependencies and model artifacts retain their own licenses and terms.
