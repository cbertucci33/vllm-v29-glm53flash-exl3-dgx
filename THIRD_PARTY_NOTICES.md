# Third-party notices

This repository is based on vLLM and credits imported work to its source
projects below. This file records the source revisions required by the
GLM-5.3 Flash runner. It does not redistribute model weights, third-party
source archives, dependency wheels, or prebuilt binary artifacts.

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

- Source: <https://github.com/local-inference-lab/sparkinfer>
- Required source commit: `d4438d490691f79022fdfc8149e1c5f161d15445`
- Relevant upstream work: PR #49
- License: Apache-2.0

Rank-sliced EXL3 uses the planned Trellis interface from this revision:
`prepare_weights`, `Caps`, `plan`, `scratch_specs`, `bind`, and `run`.

## B12X

- Source: <https://github.com/lukealonso/b12x>
- Required source commit: `ab6eea89b5b5e334ac6e9f2c503c1de60c3f216c`
- Release: `1.2.6`
- License: Apache-2.0

Only the dense MXFP8 linear path is selected for the GLM DFlash workload.
B12X sparse attention and the older expanded K-pool design are excluded.

## GLM chat template

- Source: <https://huggingface.co/zai-org/GLM-5.3-Flash>
- Model revision: `690b705278a3a58e538fcb37c2ca8b5f9511213c`
- Official template SHA-256: `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5`

The packaged template retains the model's tool, multimodal, and reasoning
protocol while adding explicit request-level thinking control and ordered
parallel tool-result handling. Its local provenance is documented under
`templates/README.md`.

## Entrpi GLM and DFlash integration

- Source: <https://github.com/Entrpi/vllm-glm-5.3-flash-spark>
- License: Apache-2.0

The runner imports GLM DFlash2, compact MTP prefill, fixed-ring,
cache-history, and runtime corrections from this source. It does not import
Entrpi's B12X sparse-MLA backend, SM90-on-SM121 target-attention route, or
expanded physical K-pool design.

## EXL3 and DFlash2 model artifacts

- EXL3 source format: ExLlamaV3 project, MIT license
- ExLlamaV3 extension source: <https://github.com/turboderp-org/exllamav3>
- Required extension source commit: `c5d9c657966ffeeaa9353f0cc899f18629da4a13`
- DFlash2 checkpoint artifacts: CC BY-NC-ND

Do not add model weights, quantized payloads, checkpoints, or derived binary
artifacts to this repository without separately confirming redistribution terms.
