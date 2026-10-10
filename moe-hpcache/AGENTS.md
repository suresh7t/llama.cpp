# moe-hpcache notes for coding agents

These notes apply to work in `moe-hpcache/`. Follow the repository root `AGENTS.md` and `CONTRIBUTING.md` first. This file records the current fork state and the source paths relevant to the proposed design.

## Current state

- The `moe-hpcache` branch adds `hp-design.md` and `pagecache.html` to upstream llama.cpp. It does not implement an HP scheduler or a MoE page cache.
- `hp-design.md` proposes heterogeneous MoE execution across existing ggml backends. The policy layer would choose work placement; ggml backends would execute it.
- `pagecache.html` describes GGBufMoE, a proposed standalone GGUF page-buffer prototype with an expert index, bounded host cache, asynchronous reads, and prefetch. Its source tree and API are proposals, not existing code in this fork.
- Treat model dimensions, page size, and performance claims in the design documents as assumptions to verify against an actual GGUF and measurements.

## Relevant source paths

- `src/llama.cpp` starts model loading and calls model metadata and tensor loading.
- `src/llama-model-loader.cpp` reads GGUF tensor descriptors and data, and handles mmap and other load modes.
- `src/llama-model.cpp` assigns layers and weight buffers to devices during model loading.
- `src/llama-graph.cpp` builds MoE routing, selected-expert matrix operations, weighting, and output aggregation in `llm_graph_context::build_moe_ffn`.
- `src/llama-context.cpp` builds and reserves graphs, configures the backend scheduler, and submits graphs for execution.
- `ggml/src/ggml-backend.cpp` assigns graph nodes to backends, splits graphs, copies cross-backend inputs, and computes the splits.

## Constraints to check before implementation

- Expert IDs are graph values produced by routing at execution time. Graph construction alone does not reveal which experts a token will use.
- The current scheduler can transfer only selected MoE experts when host-resident weights feed a `GGML_OP_MUL_MAT_ID` split. Inspect this path before adding another transfer mechanism.
- Weight buffer location influences backend assignment. Model loading currently places repeating layers by device, so per-expert placement needs a deliberate representation.
- `ggml_backend_sched_graph_compute_async` submits backend splits in order. Asynchronous submission alone does not establish concurrent execution of independent CPU, CUDA, and Vulkan work. Check dependencies, copies, buffer reuse, and synchronization before claiming overlap.
- Keep expert residency, backend execution, and GGUF storage as separate responsibilities. Define the lifetime of a page reference through any asynchronous backend use before allowing eviction.
- Derive expert-to-file mappings from real GGUF tensor descriptors. Do not assume expert tensors are contiguous or that an expert fits one page.

Read the relevant source and `skills/code-review/SKILL.md` before changing code. For a new design or implementation, confirm the contributor understands the choices and can explain them independently, as required by the root `AGENTS.md`.

