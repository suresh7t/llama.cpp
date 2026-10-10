# Strata and HPCache calibration: Qwen3.8 at 128K

This is a capacity estimate for the Qwen3.8 Flash Next IQ3_XXS model on the 8 GiB RTX 3060 Ti. Strata's recorded run uses a 32K context and INT8 KV. The 128K `q5_0` case below is a projection, not a Strata measurement; Strata does not currently offer `q5_0` KV.

## 128K context memory

The model has 12 full-attention layers, 2 KV heads per layer, and 256 values per head. A `q5_0` block stores 32 values in 22 bytes. For K and V at 131072 tokens:

```text
12 layers x 2 (K and V) x 2 heads x 256 values x 131072 tokens x 22/32 bytes
    = 1,107,296,256 bytes = 1.03125 GiB
```

This is the attention K/V tensor storage only. The current HPCache calibrator measured 2,886,282,240 bytes (2.688 GiB) of full context memory at 262144 tokens with `-ctk q5_0 -ctv q5_0`. Scaling that measurement to 131072 tokens gives about **1.344 GiB** for capacity planning when all context state is on the GPU. The measured context includes other model state, such as the attention indexer. A 128K probe is needed to replace the scaled estimate.

Strata's streaming path holds the authoritative full K/V in pinned CPU RAM and a bounded set of pages in VRAM. Its INT8 QSA page holds four cells of one layer, K and V together, in 4,224 bytes. A GPU miss fills that page from CPU RAM; writes update the CPU copy and the resident GPU page. The 4,224-byte size is specific to Strata's INT8 layout and does not describe `q5_0` storage in ggml. The HP-cache GPU allocation must be measured from its own K/V representation and chosen resident-page capacity.

## Expert-cache budget

For the proposed HPCache policy, CUDA0 gives **80% of total VRAM** to dense weights, GPU-resident context state, and expert weights. Strata loads about **3.2 GiB of dense weights** on the GPU. If the whole projected 128K context were resident on the GPU, the nominal expert-cache budget would be:

```text
expert cache = 0.80 x 8.0 GiB - 3.2 GiB dense weights - 1.344 GiB context
             = 1.856 GiB (about 1.9 GiB)
```

The **1.856 GiB is a full-residency planning figure**, not the budget for streaming KV. With GPU KV paging, subtract only the GPU page pool plus indexer and other GPU context state; the full K/V cost belongs to CPU RAM. The other 20% of VRAM must cover compute buffers and any safety margin; if they do not fit there, the expert cache must be smaller. Strata sizes its cache from live free VRAM after loading those components. Its recorded 32K/INT8 run had 1.77 GiB free at that point, reserved 700 MiB plus 184 MiB for the draft head, and initially budgeted 419 slots. Variable-size expert blobs let 549 profiled experts occupy 0.91 GiB. The 39.97 GiB expert arena stays in host memory; the 26.82 GiB PLE table is read separately.

For an illustrative 32K-cell GPU resident pool at 128K, the raw `q5_0` K/V payload would be 0.258 GiB, while the full 1.031 GiB payload stays in CPU RAM. If the 0.313 GiB difference between the projected full-context allocation and raw K/V remains on the GPU, the 80% formula gives about **2.63 GiB** for experts before page-table, transfer, and compute reserves. This is arithmetic under stated assumptions, not a measured allocation or an implemented `q5_0` paging path.

Strata's 32K/INT8 KV and indexer cost is about 0.398 GiB from its geometry. Replacing it with the projected 128K/`q5_0` full context adds about 0.946 GiB. If all other allocations stayed the same and streaming were disabled, the recorded 1.77 GiB free would fall to about 0.82 GiB, below the roughly 0.86 GiB combined reserve. This does not predict the paged-KV expert budget: that requires a chosen GPU resident-page count and measured GPU indexer, page-table, and compute allocations. Strata has no `q5_0` path, and its allocations can change with context length.

The HPCache calibrator now accepts `-ctx`, `-ctk`, and `-ctv`, uses exactly 80% of total device memory as the cache pool, reserves the full measured KV cache on CPU and 50% on each GPU, then subtracts the dense weights to report expert capacity. Free-memory and compute estimates remain diagnostics and do not reduce the owned pool. The reported result is a capacity estimate; the runtime must measure actual allocations before committing the placement.

Sources: `build-hpcache-cuda/qwen3.8-q5-calibration.json`; Strata's `strata-iq3_xxs.json`, `strata-iq3_xxs.log`, `include/strata/plan/plan.hpp`, and `src/program/generate.cpp` in `C:/Users/suresh/devspace/github/Strata`.

## Calibration benchmark method

The calibrator does not run the model router or observe expert IDs from a token. It reads `<architecture>.expert_used_count` from GGUF metadata for the layer containing the selected representative expert tensor. For Qwen3.8 this reports 10 active experts per routed token. The active count is a model configuration value, not a list of the experts selected by a particular input.

The tool selects the first suitable 3D MoE tensor, currently `blk.0.ffn_down_exps.weight`, and loads one expert slice from the GGUF file. It builds a GGML matrix multiplication using that expert's stored quantization and one F32 activation row. Each eligible backend receives the same representative operation. The tool uploads the weight, performs a warmup, then runs seven timed operations. It reports the average expert time and separately records input/output transfer timing. Transfer timing is diagnostic in the current benchmark and is not used in the split decision.

The initial active-work split assigns the configured active expert count to eligible backends using the projected cost `(assigned_experts + 1) * measured_ms_per_expert`, subject to each backend's capacity. The tool then tries up to four rounds of moving one assignment between backends. Each candidate split is measured three times; the median wall-clock time is kept only when it improves the current result by at least 5%. Assigned backends start together, so `parallel_sample_ms` is the wall-clock time of the concurrent synthetic work rather than the sum of individual expert times.

`benchmark_active_experts` describes this synthetic active-work split. It is not an expert residency map and it does not identify routed expert IDs. Residency is determined separately from `expert_cache_budget_bytes` and `max_resident_expert_groups`: the GPU owner keeps as many ranked expert weights as fit, while CPU RAM retains the complete expert set and handles GPU misses. CUDA and Vulkan entries with the same PCI device ID are backend views of one physical device; only the selected `capacity_owner` contributes to capacity and active-work assignment.
