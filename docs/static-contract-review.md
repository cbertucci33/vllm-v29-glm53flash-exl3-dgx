# GLM-5.3 Flash static interface review

Base: vLLM `98dff2a81d747d1dba01a47f939f48c3526d4206`

## Result

The review found one caller and callee mismatch in the selected runner path.
The GLM FlashKDA prefill caller omitted the required `out`, `final_state`, and
`workspace` arguments. The caller now passes the buffers it allocates.

No other argument mismatch was found in the selected path. Focused hardware
tests cover behavior that static analysis cannot establish.

## Method

- Diffed the branch against the pinned vLLM base commit.
- Traversed 93 changed Python runtime files.
- Bound 2,644 statically resolvable local and imported calls to their exact
  definitions.
- Manually resolved 18 dynamic call boundaries used by the runner.
- Compared external calls with the implementations pinned by the build.
- Checked native operator calls against registered schemas and exported
  symbols.

## Interface ledger

| Boundary | Contract checked |
| --- | --- |
| CLI to runner | Supplied warmup arguments match the selected runner signature. |
| Registry to GLM model | Registration, constructors, and forward methods agree. |
| Checkpoint to model | Rank-sliced EXL3 and fused tensor loaders receive the expected identifiers and shapes. |
| EXL3 to Sparkinfer | Planning, binding, scratch, and execution calls match the pinned Sparkinfer interfaces. |
| MXFP8 to B12X | Weight packing, matrix multiplication, and capability checks match the pinned package. |
| GLM KDA to FlashKDA | Output, final-state, and workspace buffers match the native API. |
| Target attention to FlashInfer | Cache geometry, page tables, sequence lengths, scales, and TopK capacity match the pinned decode API. |
| Cache ownership | Target cache and the DFlash ring use separate formats, storage, and accounting. |
| Scheduler lifecycle | Allocation, copy, zeroing, retirement, prefix resume, and ring ownership use the same cache classification. |
| Warmup | Autotune and warmup arguments match the installed native dependencies. |
| Chat serving | The chat template and GLM tool and reasoning parsers use matching syntax. |

## Dynamic boundaries

- Parameter-specific weight loaders are attached during model construction and
  were exercised by a complete rank-sliced checkpoint load.
- CUDA kernel correctness is covered by exact-shape tests and a full runner
  launch.
- Hardware and software paths outside the selected NVIDIA runner remain
  outside this review.

## Regression guard

`tests/models/test_glm5next_flashkda_contract.py` resolves the shared
`_flashkda_prefill` definition and requires the GLM caller to supply every
required parameter.
