# Baselines

Reference measurements for moe-hpcache. Record new runs here with date, command and conditions.

## Machine

- RTX 3060 Ti 8 GiB (CUDA0, PCIe 4.0 x16), display on GTX 1050 (not used for compute), Windows 11.
- Model: Qwen3.8-Flash-Next-GSQ-RCO IQ3_XXS (70.62 GiB, 48 MoE layers, 512 experts, 10 used).

## Strata (reference engine), 2026-10-11

Prompt: 25 tokens (chat template, "short story about a lighthouse keeper"), 128 greedy new tokens, single sequence,
CUDA0 only, `--max-context 32768 --kv int8 --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto`.
Launch: `engine\strata.exe --pack ...\packs\iq3_xxs --native <shard1> --ple-gguf <shard2> ... --tokens-file <ids>`.
Native packs refuse `--spec 0`, so "no drafts" is `--spec 2 --suffix-draft 0` with no `--mtp`.

| Mode | Flags | Decode | Per token | Notes |
|---|---|---|---|---|
| No drafts | `--spec 2 --suffix-draft 0` | **21.74 t/s** | 46 ms | 0 of 127 drafts; CPU pool 19.8 ms/round, 4.64 distinct / 5.01 routed CPU experts per layer; 1162-slot VRAM tier; pool 20.5 GB/s |
| MTP speculative | `--spec 4 --spec-min-p 0.5 --mtp ...\mtp\rt` | **29.70 t/s** | 33.7 ms | 55 rounds of 6, 74 of 128 drafts accepted (0.578), 2.35 tokens/round; CPU pool 40.4 ms/round, 9.36 distinct / 13.76 routed per layer; 1018 MiB MTP VRAM |

Prefill 24 tokens: ~34 t/s (both). Logs: kept in the session scratchpad only; re-run to reproduce.

Observations:
- Without drafts, about half of the routed experts per layer are served from the VRAM tier (profile seeded plus adaptive
  swaps every 4 rounds); the CPU expert pool runs at ~20 GB/s.
- MTP gain over no drafts is 1.37x here (acceptance 0.578 on this prompt; Strata's own log showed ~0.75).
- Replies differ slightly between modes (GPU-cache rounding); token ids diverge after ~30 tokens.

## This work (llama.cpp + hpcache provider), 2026-10-11

Same machine, `-c 65536 -ctk q5_0 -ctv q5_0 --device CUDA0`, plan `qwen-64k-q5kv-80pct.json`. All routed experts on CPU.

| Mode | Decode |
|---|---|
| Stock llama.cpp, single GPU | ~7.6 t/s |
| Placement only (`doorbell=0`) | ~8.2 t/s |
| Doorbell decode (stage 4a) | ~8.3 t/s (8.1-8.8 at 8-12 threads; ~120 ms/step, CPU experts ~83 ms) |

### Stage 4b: GPU expert pool, pinned cores, CUDA graphs (2026-10-11)

Same setup, lighthouse prompt, 200 tokens, Strata profile as the seed, 787 pool slots (1404 MiB at reserve 1280 MiB),
12 swaps per token, `pin=1` (one logical processor per physical core, process affinity).

| Threads | Cores pinned | Decode | Step | CPU experts |
|---|---|---|---|---|
| 4 | 6 | 10.9 t/s | 89 ms | 67 ms |
| 5 | 7 | 12.2 t/s | 79 ms | 57 ms |
| **6** | 8 | **13.2 t/s** | 73 ms | 52 ms |
| 8 | 8 (no spare core) | 8.8 t/s | 109 ms | 88 ms |
| 6, `pin=0` | none | 11.3 t/s | 85 ms | 64 ms |
| 6, CUDA graphs off | 8 | 11.1 t/s | 87 ms | 53 ms |

Pool hit rate 32% of routed experts (static profile alone 5-17% on these prompts). Offline replay of the routing
traces (sim of the same policy, 787 slots): 41-43% at 8-16 swaps/token; 1162 slots: 47-50%.
Goal: 20 t/s (internal). Output text is identical with graphs on/off.

Warm start (2026-10-11): expert tensors are file-backed pages, so a short run left the working set at ~29 GiB and
the first touch of each expert was a disk read inside decode. The provider now reads all expert pages at load
(`prefault=1`, default; 40 GiB in 5.7 s with a warm page cache, working set ~45 GiB, like Strata's 48 GiB).
`-t 6`, lighthouse prompt, 400 tokens: cold 14.3 t/s (66.4 ms/step), prefaulted **15.2 t/s** (62.7 ms/step, CPU experts
42.5 ms, pool hit rate 38.9%). Use prefaulted numbers from here on.

### Stage 4c: own CPU expert kernel (2026-10-11)

Same setup (prefaulted, `-t 6`, lighthouse prompt, 200 tokens). New `hpcache-cpu` kernel: one dispatch per layer on
a small pinned spin pool, ggml `vec_dot`/`from_float` per type, rows streamed in blocks with software prefetch,
and an AVX2 (AVX512-VNNI 256-bit when present) q2_0 x q8_0 dot, because ggml has only a scalar loop for q2_0.

| Kernel | Decode | Step | CPU experts |
|---|---|---|---|
| ggml graph per layer (`kernel=ggml`) | 14.1-15.1 t/s | 63-68 ms | 43-47 ms |
| own kernel, scalar q2_0 dot | 14.3 t/s | 67 ms | 47 ms |
| own kernel, AVX2 q2_0 dot | 19.2 t/s | 48.6 ms | 28.3 ms |
| **own kernel, AVX512-VNNI q2_0 dot** | **20.2 t/s** | **46.3 ms** | **26.0 ms** |

Per layer run (about 6 experts): gate/up 390 us, quantize <1 us, down 149 us. Layer 3 types: gate/up iq2_xxs,
down q2_0, n_embd 2560, n_ff 640. q2_0 dot: 0.208 -> 0.047 us per row. Own kernel vs ggml graph on the same
inputs: relative L2 difference mean 2.2e-6 (`HPCACHE_VERIFY=1`); the text still diverges after ~180 characters,
as expected from float summation order. Internal goal of 20 t/s reached (single prompt, 200 tokens).

## Planned validation prompt (not run yet)

Long multi-turn code task: ask for an LRU cache in C; feed the answer back asking to optimize the hash map, make it
SIMD and concurrent; repeat 5 times, each reply a detailed explanation with an example. Use to compare long-context
quality and speed of the provider path against stock and Strata.
