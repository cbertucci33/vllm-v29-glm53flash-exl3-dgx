# Reproducible image build

This directory defines the sources and tools used to build the GLM-5.3 Flash
runner image. Third-party source and build wheels are downloaded separately at
the revisions in `versions.env`; they are not stored in this repository.

## Inputs

- this repository at the selected release commit;
- FlashInfer with its pinned CCCL, CUTLASS, and spdlog submodules;
- Sparkinfer;
- ExLlamaV3;
- B12X;
- CUTLASS C++ source;
- the pinned vLLM runtime image;
- the pinned CUDA development image used for compilation.

`prepare_sources.sh` verifies each source checkout before creating source
archives and hashes. `download_build_deps.sh` downloads the pinned build-tool
wheels and records their hashes. The compilation stage can then run without
network access.

## Build layout

`Dockerfile.builder` combines the official runtime's Python and Torch stack
with CUDA development headers. Those headers are used only during compilation.
`Dockerfile` starts the final stage from the clean official runtime image.

The build recompiles the changed `_C_stable_libtorch` extension and overlays
the Python package plus required native artifacts. Unchanged vLLM native
extensions, including the Rust frontend, remain from the official base image.

Use `build_wheels.sh` for external wheels and
`build_core_extension.sh` for the vLLM CUDA extension. Combine the outputs and
run `finalize_artifacts.sh` to produce the hash manifest. Then create the final
Docker context:

```bash
python3 build/prepare_image_context.py \
  /path/to/source-archives \
  /path/to/wheelhouse \
  /path/to/image-context
```

The output directory must be empty. The script verifies both input manifests
and writes the provenance document consumed by `Dockerfile`.

## Compatibility patches

The pinned FlashInfer source requires CUTLASS DSL 4.7.0 or newer. The
Sparkinfer and B12X patches update package constraints for CUTLASS DSL 4.7.0.
The B12X patch also corrects the MXFP8 capability predicate when the kernel
returns `(False, reason)`.

The ExLlamaV3 patch excludes optional x86 CPU all-reduce translation units on
ARM64 and supplies fail-closed stubs. It does not change the CUDA EXL3 kernels.

## Reproducibility rules

- Build from clean source checkouts at the pinned commits.
- Hash source archives, wheels, extensions, and the toolchain.
- Compile external dependencies after the download step with network access
  disabled.
- Use the architecture flags in `versions.env` for GB10.
- Install local wheels with `--no-deps`.
- Adjust build parallelism to available RAM; the values in `versions.env` are
  conservative defaults, not runtime settings.
- Verify imports, native symbols, packaged revisions, and artifact hashes in
  the final image.

The final image writes its provenance to
`/opt/glm53-runner-provenance.json` and its verified runtime versions to
`/opt/glm53-runner-versions.json`.
