# HPCache calibration prototype

See [Strata and HPCache calibration](../../docs/strata-hpcache-calibration.md) for the Qwen3.8 128K `q5_0` capacity estimate.

Build `llama-hpcache-calibrate` with the backends you want to measure, then run:

```sh
llama-hpcache-calibrate -m model.gguf -ctx 128kib -ctk q5_0 -ctv q5_0 -o hpcache-calibration.json
```

Pass `-ctx 128kib` to measure a 128K-token context. The suffixes `k`, `ki`, `kib`, `m`, `mi`, and `mib` are accepted; values without a suffix are token counts. Pass `-ctk q5_0 -ctv q5_0` to measure 5-bit K and V caches. Quantized cache types enable Flash Attention in the memory probe. Both types default to F16.

The tool reads the first 3D MoE expert weight tensor from a GGUF model. For split models, pass the first shard to `-m`. It measures one expert matrix operation using the tensor's stored quantization on each available backend. Each timed operation copies its input to the backend and its output back to host memory. The tool assigns the model's active expert count across devices using those timings, compares nearby splits with concurrent runs, and writes the measurements and proposed work split to JSON.

The tool estimates context and compute buffers at the requested context length, with one sequence, a 2048 token batch, and a 512 token microbatch. It probes all layers on CPU, then all layers on each GPU separately, so each backend has a full-context estimate. The calibration policy owns exactly 80% of total memory for caching. Free-memory and compute estimates are reported for diagnostics but do not reduce this owned pool. CPU reserves the full measured KV cache as the authoritative copy. Each GPU reserves a fraction of its measured KV cache (`-kvf`, default 1.0, so the full KV cache stays on the GPU; use 0.5 for a paged-KV plan), keeps a full dense-weight copy, and reports the remaining pool as expert-cache capacity. The token embedding table is not counted as a GPU dense weight because input lookups run on the CPU, unless the model has no separate output head. A backend is excluded if its owned pool cannot fit dense weights plus its KV reservation. This is a capacity estimate, not an allocation test.

The JSON separates expert weights, the `per_layer_token_embd.weight` PLE table, and other (dense) weights. It reports P95 and P99 stored sizes for dense tensors, dense blocks, and per-layer experts. Each device's `expert_cache_budget_bytes` subtracts the dense copy and that device's KV reservation from its usable 80% pool; the PLE table is offloaded. The mean and P99 slot counts are capacity estimates, not selected experts or a measured placement. Independent full-layer probes measure working memory; a future split execution graph can require different buffers. `max_resident_expert_groups` retains the earlier single-layer estimate. If combined expert capacity is less than the expert weights, the tool writes the shortfall to JSON and exits with an error. The JSON describes an active expert work split, not a map of individual expert weights. No llama.cpp loading or scheduling behavior uses the JSON yet.

The measurement uses one matrix from one layer; a full MoE layer has other operations and synchronization costs. Compare the proposed split with full model decoding before using it as a placement policy.

CUDA and Vulkan entries with the same PCI device ID are two backend views of one physical device. The calibration JSON reports both measurements, but capacity and active-work assignment use one owner backend per physical device and do not add the two memory pools together. `benchmark_active_experts` describes the measured active-work split; it is not an expert residency map.
