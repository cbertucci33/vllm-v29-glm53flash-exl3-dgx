# Third-party notices

This repository is based on vLLM and retains imported work with its original
Git authorship. This file records the source revisions required by the current
GLM-5.3 Flash v2 path. It does not redistribute model weights or prebuilt
binary artifacts.

## vLLM

- Source: <https://github.com/vllm-project/vllm>
- Base: `v0.29.0`, commit `98dff2a81d747d1dba01a47f939f48c3526d4206`
- License: Apache-2.0
- Imported GLM model support: vLLM PR #53906
- Imported native GLM sparse-MLA fixes: vLLM PR #55277

## FlashInfer

- Source: <https://github.com/flashinfer-ai/flashinfer>
- Required source commit: `5cc867a9bb560bc89b91dbc738a9f63f09beb89b`
- Relevant upstream work: PR #4802 and PR #4947
- License: Apache-2.0

The pinned source provides the native SM120 and SM121 `GLM53_NOPE` sparse-MLA
kernel. Build the runner against this exact revision or a reviewed replacement.

## Sparkinfer

- Source: <https://github.com/tonyd2wild/sparkinfer>
- Required source commit: `d4438d490691f79022fdfc8149e1c5f161d15445`
- Relevant upstream work: PR #49
- License: Apache-2.0

Rank-sliced EXL3 uses the planned Trellis interface from this revision:
`prepare_weights`, `Caps`, `plan`, `scratch_specs`, `bind`, and `run`.

## EXL3 and DFlash2 model artifacts

- EXL3 source format: ExLlamaV3 project, MIT license
- DFlash2 checkpoint artifacts: CC BY-NC-ND

Do not add model weights, quantized payloads, checkpoints, or derived binary
artifacts to this repository without separately confirming redistribution terms.
