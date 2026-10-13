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

## Standalone tests and scratch scripts

Everything that is not part of the fork (micro benchmarks, simulators, run and sweep scripts, test prompts, traces,
logs) lives outside the repository in `C:\Users\suresh\devspace\claude`, one folder per tool or study. Do not put
such files in the repository, in the system temp directory, or in the session scratchpad. Index: `README.md` there.

- `mbw`: CMake project for the memory bandwidth tool (`membw.cpp`); build in a new directory, `b` holds a stale cache.
- `hpcache-bench`: run and sweep scripts (`run3.ps1`, `kvtype.ps1`, `kvx.ps1`, `ubsweep.ps1`, `spare_ab.ps1`), the
  admission simulators (`sim2.py` to `sim4.py`, `sw4.py`; run from that folder), the 16K test prompt, routing traces,
  and `logs/` with the `.out`/`.err` of earlier runs.

## Expert arena (hpcache-arena)

The provider reads the expert tensors from the GGUF itself (unbuffered, 2 MiB requests) into one page-locked region and
repoints the expert tensors' `data` at it in `make_layer`. The core's mapping is not used for experts afterwards. Keep
the pointers valid for the life of the decoder, and do not register host memory with a second GPU context alive (limit
about 25 GiB, then CUDA allocations fail). Numbers: BASELINES.md, "Expert arena in the provider".

## GPU path for missed experts

The decode graph of a layer fetches a share of the missed experts from the page-locked arena into device staging
slots (kernel read through PCIe) and computes them on the GPU while the CPU computes the others. The device and the host
must apply the same selection rule (see BASELINES.md, "GPU path for missed experts"); change both together. Staging
slots are shared by all layers and are safe because a layer's staged compute precedes the next layer's publish on the
stream. Unused slots must be zeroed.

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
- M1b: paged KV (CPU authoritative copy, bounded GPU page pool) for long context, only when contexts of 128K and more
  are needed. Compressed KV types such as TurboQuant are dropped: the speed comes from the expert path, not the KV.
- M2: Vulkan and multi-backend in one layer, and batched decode (verify windows for MTP speculative decoding, which the
  base already has: the largest remaining decode lever, Strata 21.7 -> 29.7 t/s).
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
   Stage 4d done (BASELINES.md): swaps queued to a copier thread and mapped one step later, `max_swaps` default 24:
   20.2 t/s at `-c 65536 -ctk q5_0`, 22.0 t/s at `-c 32768 -ctk q8_0 -ub 128` with `reserve_mib=700` (Strata 21.7).
   Rejected with data: iq2_xxs AVX512 gather / mask-sign dots (slower), prefetch distance and hint changes (no
   effect), pinning the expert tensors in place (driver limit about 27 GiB).
   Reserve is now derived from `-ub` (`450 + 1.6 * n_ubatch` MiB) unless `reserve_mib` is given.
   Stage 4e done: q2_0 down dot on 256-bit VBMI/VNNI with per-activation side data (0.046 -> 0.022 us/row):
   21.9-22.0 t/s at `-c 32768 -ctk q8_0 -ub 128 reserve_mib=700`, 128 tokens (Strata no drafts 21.74).
   Long-session profile (BASELINES.md, 2026-10-12): 31.7K tokens through llama-server at 17.3 t/s average; the only
   thing that degrades is GPU-side "other" time, +0.36 ms per 1000 tokens of context (linear), about 13x the K/V
   bandwidth floor. Next: compare KV types (q5_0/q8_0/q4_0/f16) at 16K context to see whether the cost is the
   single-token FlashAttention dequant (core kernel) or the graph around it; scratchpad `kvtype.ps1` + `long_prompt.txt`.
   - Open: a spare slot per layer so a replaced expert keeps serving while its copy is in flight (about 4-5 points of
     hit rate); multi-prompt validation (single-prompt numbers move by about 1 t/s with the slot count); the gate/up phase (394 us/layer) is the largest remaining CPU part.
   - Validate on the long LRU-cache prompts and a second layer mix (other types may also lack an AVX2 dot).
   - gate/up phase (390 us/layer) is the largest CPU part but near its load-port limit (see BASELINES.md); more pool slots
     (compute-buffer-aware reserve from `-ub`, calibrator fix).
   - q2_0 AVX2 dot could be contributed to ggml-cpu arch/x86/quants.c; it lives in the provider for now.
   - GPU expert pool (stage 4b): profile-ranked admission, used-bytes counter, +/-10% plan tolerance, live-free-memory guard.
   - Replay-safe custom ops so CUDA graphs stay enabled.
   - Overlap the shared expert with the CPU experts.
4. Calibrator: account for the real compute buffer (1061 MiB observed vs 509 MiB estimated).
5. M1b paged KV.
6. Cleanup: tracing and routing dump behind env vars or removed; update stale 50% GPU KV text in hp-design.md 17.1 and strata-hpcache-calibration.md.

Read the relevant source and `skills/code-review/SKILL.md` before changing code. For a new design or implementation, confirm the contributor understands the choices and can explain them independently, as required by the root `AGENTS.md`.

