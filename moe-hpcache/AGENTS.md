# moe-hpcache notes for coding agents

These notes apply to work in `moe-hpcache/`. Follow the repository root `AGENTS.md` and `CONTRIBUTING.md` first. This file records the current fork state and the source paths relevant to the proposed design.

## Current state

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

## Design constraints (owner directives)

These are set by the project owner. Do not change them without asking.

- MoE is pluggable. A provider supplies its MoE implementation through a shared library, loaded with `--moe-provider <name>`. `moe-hpcache` (Heterogeneous Parallel execution with a database style page cache) is one provider. Other providers may work in a different way.
- llama.cpp core provides three things to every provider:
    - inference on any ggml backend device
    - a default expert placement algorithm
    - a default scheduling algorithm
- Placement and scheduling are the abstraction interfaces being defined now. The provider may override them. If it does not, the core default is used. Keep these interfaces provider neutral: no CUDA, Vulkan, page cache, or Strata terms in the core side.
- Keep llama.cpp core changes minimal: wiring, argument handling, and small adapter code only. Policy and mechanism belong in the provider library.
- The current `llama_moe_provider` hooks (`tensor_load`, `place_tensor`, `build_dispatch`, `before_compute`, `after_compute`) are a first sketch and may be removed and rebuilt. Define the placement and scheduling interfaces first, then rebuild the provider interface on top of them. Do not extend the sketch.
- A larger later milestone refactors the core so llama.cpp itself is a reference implementation of the same interfaces, and end users supply their own capabilities as providers. Do not start this refactor early. Shape the interfaces so the refactor needs no ABI break where possible.

- Scheduling is provider driven dynamic dispatch (Strata style doorbell): routed ids are read at run time, GPU work is published first, CPU misses run concurrently, and the merge waits for both. Staying inside the ggml graph with a remap table was measured at about 6-7 t/s against about 25 t/s for Strata, so it is not acceptable as the design. Strata source is at `C:\Users\suresh\devspace\github\Strata`.
- The ggml custom op with a backend stream disables CUDA graph reuse for a graph that contains it. This is accepted for now and is benchmarked once the doorbell path exists.
- The calibration JSON is a plan. The provider sizes the GPU expert pool at run time from a used-bytes counter against the plan budget and live free device memory, and the plan budget has a plus or minus 10% tolerance.
- Prefill (n_tokens > 1) always uses the stock `build_moe_ffn` path. The provider schedule hook applies to single-token decode only in M1.
- KV update is core driven: one `kv_commit(seq, cell_begin, n_cells)` call per ubatch after compute. The provider owns the authoritative CPU copy.

## Milestones

- M1a: CPU + CUDA expert dispatch and merge, single-sequence decode, stock GPU KV cache, on a context where full KV fits.
- M1b: paged KV (CPU authoritative copy, bounded GPU page pool) for long context.
- M2: Vulkan and multi-backend in one layer, and batched decode.
- M3: reference-implementation refactor. Merge `include/llama-moe-provider.h` (C ABI) and `common/moe-provider.{h,cpp}` (loader):
  move library loading into libllama behind a small C API (open by name and args, close), so `llama_model_params` carries
  the name and args instead of a pointer, the loader class in `common/` disappears, and the stock path becomes the first
  provider behind the same ABI.

## Next steps (temporary, 2026-10-11; delete when resolved)

State: stage 4a doorbell decode works and is verified (8.3 t/s vs stock 7.6, placement-only 8.2). CPU experts are DRAM-bound (~46 ms/step floor of ~120 ms). Strata's 30-34 t/s used `--spec 4 --mtp` (~75% acceptance) and an expert profile, so it is not a non-speculative target.

1. DONE (full numbers in moe-hpcache/BASELINES.md; MTP run: 29.70 t/s): Strata with no drafts (`--spec 2 --suffix-draft 0`, no --mtp), 32K int8 KV, 128 tokens: 21.7 t/s. CPU pool 19.8 ms/token for ~5 routed experts/layer, other ~5 served from a 1162-slot VRAM tier (profile + adaptive swaps every 4 rounds), pool 20.5 GB/s. Ours: 8.3 t/s with all 10 experts on CPU at ~83 ms.
2. Direction: non-speculative 20+ t/s is reachable, so single-token work comes first. Two levers: (a) GPU expert tier with ~50% hits via profile + adaptive admission (our cross-prompt top-26 coverage of 14-21% understated this; adaptivity within a prompt matters), (b) a faster CPU expert kernel (Strata: q8_0 activation quantize, 7 workers + host thread, ~0.08 ms/expert vs ours ~0.17).
   Stage 4b done (numbers in BASELINES.md): GPU pool + adaptive swaps + CUDA graphs re-enabled + core pinning: 13.2 t/s at `-t 6`.
   Remaining gap to 20 t/s: CPU experts 52 ms (Strata ~20 ms for ~5 experts/layer, ours ~1.1 ms/layer for ~6.8) and ~21 ms other.
3. Next, in order of expected gain:
   Stage 4c done: own CPU expert kernel + AVX2/VNNI q2_0 dot: 20.2 t/s (goal reached on one prompt).
   - Validate on the long LRU-cache prompts and a second layer mix (other types may also lack an AVX2 dot).
   - gate/up phase (390 us/layer) is now the largest CPU part; more pool slots (compute-buffer-aware reserve, calibrator fix).
   - q2_0 AVX2 dot could be contributed to ggml-cpu arch/x86/quants.c; it lives in the provider for now.
   - GPU expert pool (stage 4b): profile-ranked admission, used-bytes counter, +/-10% plan tolerance, live-free-memory guard.
   - Replay-safe custom ops so CUDA graphs stay enabled.
   - Overlap the shared expert with the CPU experts.
4. Calibrator: account for the real compute buffer (1061 MiB observed vs 509 MiB estimated).
5. M1b paged KV.
6. Cleanup: tracing and routing dump behind env vars or removed; update stale 50% GPU KV text in hp-design.md 17.1 and strata-hpcache-calibration.md.

Read the relevant source and `skills/code-review/SKILL.md` before changing code. For a new design or implementation, confirm the contributor understands the choices and can explain them independently, as required by the root `AGENTS.md`.

