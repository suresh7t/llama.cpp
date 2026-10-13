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

## Like-for-like with Strata no-drafts (2026-10-11)

Same prompt as the Strata baseline (lighthouse, 128 new tokens, greedy), `-c 32768 -ctk q8_0 -ctv q8_0`, `-t 6`,
prefaulted, own kernel, Strata profile as the seed. Strata did not use `--kv-resident`, so its KV was also all in VRAM.

| Setup | Pool slots | Hit rate | CPU experts | Decode |
|---|---|---|---|---|
| `-ub 512`, reserve 1280 MiB | 871 | 31.7% | 29.5 ms/step | 18.6 t/s |
| `-ub 128`, reserve 700 MiB (2 runs) | 1197 | 35.3% | 27.5-27.8 ms/step | 19.5-19.6 t/s |
| Strata, no drafts | 1162 | about 50% | 19.8 ms/round | 21.74 t/s |

Per missed expert the CPU cost is equal (ours 89 us, Strata 82 us); the gap is the hit rate (35% vs 50% at the same
slot count), i.e. the admission policy. `-ub 128` needs about 500-550 MiB of reserve; below that the compute buffer
allocation fails (450 MiB failed, 550 MiB worked).

## CPU kernel analysis (2026-10-11)

Per layer, about 6.5 missed experts, 6 pinned threads, `HPCACHE_PHASES=1`, per thread and run (us):
wake 1 | gate/up work 397, barrier wait 9 | quantize 0.3, wait 6 | down work 148, wait 6 | reduce 2.
Barriers, imbalance and wake-up are about 4% of the layer. The work phases run 1.25-1.5x slower than the one-thread,
L1-resident dot rate (264 + 118 us ideal), consistent with all-core clocks and DRAM streaming (token working set about
480 MB against a 16 MiB L3, L2 512 KiB per core).

Tried and rejected (iq2_xxs gate/up dot, one thread, row in L1, us per row, results identical to stock):
stock AVX2 0.204; AVX512 gathers + popcount signs 0.324; AVX2 direct loads + ksigns byte mask 0.229.
Prefetch distance 2/4/8 rows (gate,up / down), T0/T1/NTA hints: gate/up work 397-414 us, no gain.
Per-type one-thread dot rate, G weights/s: iq2_xs 13.5, iq2_xxs 12.2, iq2_s 12.8, iq3_xxs 10.9, iq3_s 7.9,
q2_0 13.6 (own SIMD), iq4_nl 21.3.

## Admission policy and overlapped swaps (2026-10-11)

Offline replay (`sim2.py`, pure python, same policy as `admit()`; matches live within 1.3 points on the first 127
tokens): decay, threshold and margin move the hit rate by under 1 point; the number of swaps per step and the number
of candidates per layer drive it. Live, synchronous swaps: 12 -> 35.3% hit, 24 -> 41.4%, 48 -> 46.8%, but each swap
costs about 0.2 ms on the critical path (a copy from pageable file-mapped memory blocks its caller), 5.4 ms per step
at 48 swaps. Pinning the expert tensors in place with `cudaHostRegister` works for about 27 GiB (32 layers), then
the driver runs out of lockable memory and CUDA calls start failing, so it is not used.

Now: the swaps are queued to a copier thread and mapped one step later (`async_swaps=1`, default). Admission on the
critical path drops from 2.4-5.4 ms to 0.3-0.6 ms per step. The one-step delay costs hit rate (48 swaps: 46.8% ->
41.5%), so the best cap moved: lighthouse prompt, `-t 6`, `-c 32768 -ctk q8_0`, `-ub 128`, reserve 700, 1197 slots,
127 steps, two runs each:

| max_swaps (async) | Hit rate | Generation |
|---|---|---|
| 12 | 34.3% | 20.0-20.1 t/s |
| **24 (default)** | 43.0% | **21.1-21.2 t/s** |
| 36 | 44.1% | 21.2 t/s |
| 48 | 41.5% | 20.8 t/s |
| 48, synchronous | 46.8% | 20.1-20.4 t/s |

Defaults, 200 tokens, same prompt: `-c 65536 -ctk q5_0`, `-ub 512` (787 slots): 38.4%, 20.2 t/s;
same with `-ub 128`, `reserve_mib=700` (1113 slots): 43.1%, 20.9 t/s; `-c 32768 -ctk q8_0`, `-ub 128`,
`reserve_mib=700` (1197 slots): 47.5%, **22.0 t/s** (Strata no drafts: 21.74 t/s at 128 tokens, 1162 slots).
Open: the delayed mapping loses about 4-5 points of hit rate; a spare slot per layer would keep the replaced expert
serving during the copy.

## q2_0 down kernel, 256-bit VBMI/VNNI (2026-10-12)

The down activation is quantized once per expert with per-64-weight side data (scale of each 32-block, sum of every
4 activations), so the q2_0 dot does one `multishift` unpack, one `dpbusd` per 32 weights, an integer offset
correction and one scaled FMA. One-thread bench: 0.046 -> 0.022 us/row, relative difference to the stock dot
about 1e-6. A 512-bit version was slower in situ (down 129 us vs 119 us, gate/up 414 us vs 394 us: the core drops its
clock), so the 256-bit one is used. `HPCACHE_NO_VBMI=1` selects the previous kernel.

Per layer, per thread (us): down work 152 -> 119, gate/up unchanged (394). Step time at equal hit rate: -2 to -3 ms.
Down prefetch distance 4/8/16/32 rows: no effect. Machine read bandwidth (membw): 1 thread 22.3 GB/s, 2 30.6, 3 35.5,
4 36.7, 6 39.8, 8 41.1 GB/s; the expert path moves about 20-23 GB/s, so it is not at the DRAM limit either.

Same prompt, `-t 6`, defaults (async swaps, `max_swaps=24`):

| Setup | Slots | Hit rate | Generation |
|---|---|---|---|
| `-c 32768 -ctk q8_0`, `-ub 128`, reserve 700, 128 tokens | 1197 | 43.0% | **21.9-22.0 t/s** (Strata no drafts 21.74) |
| `-c 65536 -ctk q5_0`, `-ub 128`, reserve 700, 200 tokens | 1113 | 43.1% | 21.9 t/s |
| `-c 65536 -ctk q5_0`, `-ub 512`, default reserve, 200 tokens | 787 | 38.4% | 20.8 t/s |

## Reserve derived from `-ub` (2026-10-12)

The host now passes `n_ubatch` to the provider; without `reserve_mib` the reserve is `450 + 1.6 * n_ubatch` MiB
(measured compute buffer: 1061 MiB at `-ub 512`, allocation fails below about 500 MiB at `-ub 128`). All of
`-ub 512/256/128/64` start without tuning (878 / 1108 / 1223 / 1280 slots at `-c 32768 -ctk q8_0`).

Caution on single-prompt numbers: the GPU and CPU round differently, so changing the slot count by even 26 slots
changes which experts run where and the text diverges. Same prompt, `-ub 128`: reserve 700 (1197 slots) 43.0% hit,
22.0 t/s; reserve 654 (1223 slots) 38.5% hit, 21.1 t/s. Treat differences under about 1 t/s between configurations as
noise until several prompts are compared (the LRU-cache multi-turn prompt is the planned set).

## Long session through llama-server (2026-10-12)

`llama-server`, `-c 65536 -ctk q5_0 -ctv q5_0 -t 6 -ub 128 -np 1`, provider defaults (async swaps, `max_swaps=24`,
reserve derived from `-ub`: 1139 slots), aider, one request: 8363 prompt tokens (123 s, 67.9 t/s), 31746 generated
tokens in 1836 s = **17.29 t/s average** (57.8 ms/token), context 8.4K -> 40.1K. `HPCACHE_STATS_EVERY=250` windows
(126 windows), averaged per 3750 steps:

| Steps | Hit rate | Step | CPU experts | Other (step - CPU) | Admission |
|---|---|---|---|---|---|
| 0-3750 | 39.2% | 50.3 ms | 25.3 ms | 25.0 ms | 0.50 ms |
| 7500-11250 | 37.7% | 53.4 ms | 26.2 ms | 27.2 ms | 0.59 ms |
| 15000-18750 | 35.8% | 56.6 ms | 26.2 ms | 30.4 ms | 0.94 ms |
| 22500-26250 | 36.9% | 59.1 ms | 26.3 ms | 32.8 ms | 0.70 ms |
| 30000-31500 | 36.7% | 61.1 ms | 26.0 ms | 35.0 ms | 0.96 ms |

Least squares over all windows: "other" +0.358 ms per 1000 tokens of context (intercept 24.2 ms), linear, no
acceleration; CPU experts +0.024 ms per 1000 (flat); hit rate -0.06 points per 1000 (flat). The whole slowdown is the
GPU-side cost growing with context. K/V bandwidth alone would be about 0.03 ms per 1000 tokens (12 KiB per token at
450 GB/s), so the attention path runs at about 8% of memory bandwidth at batch 1. Extrapolated to 64K: about 47 ms
other, 73 ms step, 13.7 t/s. The q5_0 FlashAttention kernel (rebuilt with `GGML_CUDA_FA_QUANTS`) cut the slope from
about 0.94 to 0.36 ms per 1000 tokens.

perfmon (1 s samples, same period): CPU 45-48%, RAM available min 9.0 GiB, GPU memory constant 7.85 GiB, disk read
mean 0.18 MB/s (max 2.1; PLE rows), page faults median 7.3K/s with 100% steady spikes, no trend with context, so
they do not explain the slowdown (about 0.3 ms per step at most).
Reasoning: the model's own default effort is the maximum (Strata documents "high" as the default and its `chat.py`
sends `medium`); aider sends no `reasoning_effort`, so a single answer spent about 30K tokens thinking.

## -ub sweep: prefill against decode (2026-10-12)

Same 16K-token prompt, q5_0 KV, -c 32768, -t 6, reserve derived from -ub, 64 generated tokens, llama-cli.
Decode figures are the first 32-step window right after the prefill (context about 16K).

| -ub | pool slots | prefill t/s | decode step ms | decode t/s | hit rate |
|---|---|---|---|---|---|
| 128 | 1305 | 64.0 | 55.3 | 18.1 | 39.7% |
| 256 | 1209 | 98.3 | 55.8 | 17.9 | 38.3% |
| 512 | 979 | 153.5 | 57.4 | 17.4 | 35.2% |
| 1024 | 519 | 241.3 | 60.2 | 16.6 | 25.6% |

Decode cost grows because each 128 of -ub takes about 80 slots (1.6 MiB per ub token of compute buffer) from the pool.
Break-even in tokens (P prompt tokens prefilled, G tokens generated): 512 beats 128 unless G/P > 5.7;
256 beats 128 unless G/P > 17; 1024 beats 512 only when G/P < 0.9. Aider-style sessions (large prompt, a few
thousand generated tokens) favour -ub 512; -ub 128 only for pure generation with a short prompt.

## Spare slot per layer (2026-10-12): no gain, off by default

One empty slot per layer receives the copy and the victim stays mapped until it lands (`spare_slot=1`).
-ub 512, 16K prompt, 400 tokens at temp 0, cumulative pool hit rate over the decode, two runs each:

| spare_slot | slots usable | hit rate | decode t/s |
|---|---|---|---|
| 0 | 979 | 36.7% | 18.2 / 18.1 |
| 1 | 931 | 34.9% | 18.0 / 18.0 |

The 48 slots given up cost more than the victim hits retained during the copy (victims are the coldest experts, so
they rarely hit). The earlier "4-5 points" estimate for this idea was wrong. The option stays for later experiments.

## Pinned arena feasibility (2026-10-12, `devspace\claude\pinned-arena`)

One `VirtualAlloc` region (4 KiB pages), registered with `cudaHostRegister` (portable + mapped) in 256-512 MiB slices,
64 GB machine, nothing else running:

| Setup | Result |
|---|---|
| 40 GiB, default working set | all registered in 0.23 s; H2D async 1.75 MiB chunks 24.6 GB/s; GPU zero-copy read 24.5 GB/s; CPU read 42-46 GB/s with 4-8 threads |
| 48 GiB | registration stops at 46.5 GiB (available RAM 7 GiB) |
| 40 GiB with `SetProcessWorkingSetSizeEx` hard minimum | **breaks**: copies fail with out of memory from about 18-28 GiB registered. Do not set it. |
| `cudaHostAlloc` in 512 MiB pieces | fails at 19.5 GiB, and CUDA stops working afterwards |
| mapped flag off | no difference in the limit |

A copy must stay inside one registered slice (a chunk that straddles two slices fails with invalid argument), so
expert slots are laid out per layer inside slices. Large pages were not tested (the lock-pages privilege is not set).

Second CUDA context (found while moving the probe into the calibration tool): with a context on the GTX 1050 present in
the process, `cudaHostRegister` of 40 GiB fails at about 25 GiB (or succeeds read-only and then every copy fails with out
of memory), whatever the flags. Registering first, or hiding the card (`CUDA_VISIBLE_DEVICES=0`), gives the full 40 GiB.
`llama-hpcache-calibrate` now runs the probe before the backends load and writes `host_arena` to the JSON (40.00 GiB
registered, 24.6 GB/s). The provider must register its arena before anything touches the second GPU, or the run needs
`CUDA_VISIBLE_DEVICES`.

## NVMe load rate (2026-10-12, `devspace\claude\nvme-load`)

`D:` is a WD_BLACK SN750 SE NVMe. Unbuffered overlapped reads (no page cache) of 4 GiB of the first GGUF shard, one
thread, GB/s by request size and requests in flight:

| Request | depth 1 | 4 | 16 | 32 |
|---|---|---|---|---|
| 4 KiB | 0.10 | | | 0.50 |
| 8 KiB | 0.18 | | | 0.71 |
| 64 KiB | 0.50 | | | 2.05 |
| 128 KiB | 0.46 | 1.75 | 3.45 | 3.47 |
| 256 KiB | 0.59 | | | 3.47 (depth 8) |
| 1 MiB | 1.77 | 3.47 | 3.47 | 3.47 |
| 4 MiB | 2.73 | 3.46 | 3.47 | 3.47 |
| 8 MiB | 2.99 | 3.47 | 3.47 | 3.47 |

The device saturates at 3.47 GB/s with 256 KiB-1 MiB requests and 4-8 in flight, so 40 GiB of experts loads in about
12 s cold. 4 KiB and 8 KiB requests are latency bound (0.1-0.7 GB/s): there is nothing to gain from an 8 KiB unit.
The arena loader should issue 1-4 MiB reads, 8 in flight, straight into the registered region.

Deeper queues (completion port, up to 512 requests in flight; 2 GiB window, GB/s):

| Request | depth 32 | 64 | 128 | 512 | 128 deep, 2 threads | 4 threads |
|---|---|---|---|---|---|---|
| 4 KiB | 0.55 | 0.58 | 0.62 | 0.63 | 1.06 | 1.13 |
| 8 KiB | 0.73 | 1.20 | 1.23 | 1.21 | 1.97 | 1.88 |
| 16 KiB | 1.62 | 2.31 | 2.25 | 2.38 | 2.04 | 2.06 |
| 64 KiB | 2.17 | 2.15 | 2.13 | 2.14 | | |
| 256 KiB | 3.46 | 3.46 | 3.46 | 3.45 | | |
| 1 MiB | 3.46 | 3.46 | 3.46 | 3.41 | | |

A single submitting thread is limited to about 150K requests per second (0.62 GB/s at 4 KiB). With 2-4 threads at depth
128 the device reaches about 275K 4 KiB requests per second (1.1 GB/s), a third of the sequential rate, so a 4 KiB unit
stays IOPS bound even with a deep queue; 256 KiB and larger reach 3.46 GB/s at any depth.

## Expert arena in the provider (2026-10-12)

`hpcache-arena.{h,cpp}`: the expert tensors of all shards are read from the GGUF with unbuffered 2 MiB requests (depth
4, one thread) into one 2 MiB aligned region, laid out per layer, and page-locked per layer (`cudaHostRegisterMapped`,
current device). `make_layer` points the expert tensors' `data` at the arena (after comparing the first tensor with the
model mapping at three places), so the CPU kernels, the pool copies and the stock prefill path all read from it and the
model mapping is never touched. Options: `arena=0/1` (default 1), `arena_mib`, `pin_mib`.

Load: 40.01 GiB in 14-15 s (2.9-3.1 GB/s); the old warm-cache prefault took 6 s but needs the file cache.

| Case | Decode (short prompt, `-ub` 128, 3 rounds) | Prefill (16K prompt, `-ub` 512) |
|---|---|---|
| mapping, prefaulted | step 46.0 / 46.4 / 45.5 ms, 21.2-21.6 t/s | 150.6 t/s |
| arena | step 45.3 / 45.0 / 43.7 ms, 21.2-21.7 t/s | **238.8 t/s** |

Decode is the same (hit rate identical, 42.3%); prefill is 59% faster because nothing faults or competes with a file
cache. With both GPUs visible the arena would break the device after about 25 GiB, so the provider locks at most 16 GiB
then (20 GiB tested: 20.3 t/s, 12 GiB: 19.5 t/s on the unlocked remainder) and logs how to lock all of it
(`CUDA_VISIBLE_DEVICES=0`). Unlocked layers are still read from the arena; only the copies from them are synchronous.

## GPU path for missed experts (2026-10-12)

The GPU computes a share of the experts that are not in its pool. Rule, applied identically on the device
(`pcie_select_kernel`, `pcie_stage_kernel` in `hpcache-cuda.cu`) and on the host (`serve`): take the misses in routing
order; the GPU gets the last `g = min(pcie_max, (misses * frac256) >> 8)`, the CPU the rest. The stage kernel copies those
experts from the page-locked arena (read through PCIe, one kernel, many blocks) into device staging slots shared by all
layers, zeroes the unused slots, and the same expert graph as the pool path computes them from there. Layers whose
arena range is not locked do not use it. Options: `pcie=0/1` (default 1), `pcie_max` (3), `pcie_frac` (0.55).

Two variants were tried. Reading the experts in place with the ggml expert kernels (views labelled with a device buffer,
data pointing at the mapped arena) ran at only 14 GB/s effective and gave nothing: CPU experts 21.9 -> 14.7 ms but the GPU
side grew by 9 ms (step 44.1 vs 42.5 ms). Staging first, then computing from VRAM:

| Short prompt, `-ub` 128, 192 tokens | step | CPU experts | generation |
|---|---|---|---|
| off | 42.4 ms | 21.8 ms | 21.2 t/s |
| frac 0, path on (fixed cost) | 43.1 ms | 22.0 ms | 20.8 t/s |
| frac 0.25, max 4 | 38.6 ms | 17.2 ms | 23.0 t/s |
| frac 0.55, max 4 | 39.2 ms | 14.4 ms | 23.0 t/s |
| frac 0.4, max 3 | 38.1 ms | 16.6 ms | 23.5 t/s |
| **frac 0.55, max 3** | **37.7 ms** | 13.6 ms | **23.7 t/s** |
| frac 0.7, max 5 | 40.1 ms | 10.3 ms | 22.4 t/s |

16K prompt, `-ub` 512, 128 tokens: generation 17.7 -> 19.8 t/s (+12%), prefill unchanged (238 t/s).

CPU and GPU chains are now balanced (about 280 us each per layer); the expert bytes of a layer (9.5 MiB at 46% hit) over
the two paths (20 + 24 GB/s) would be 226 us at the ideal. Two bugs on the way: stale staging slots hold another layer's
tensor type and decode to NaN (hence the zeroing), and the staging buffer needs the backend's row padding as slack.

## Share from the calibration, adjusted at run time (2026-10-12)

`llama-hpcache-calibrate` writes `expert_split` (CPU rate of the sample expert matrix, PCIe copy rate out of the locked
arena, `pcie_frac = pcie / (pcie + cpu)` clamped to 0.1-0.8, `pcie_max` = experts of the GPU path per layer). The provider
takes both from the plan unless `pcie_frac` / `pcie_max` are given. Every 16 steps the share moves by 0.03 toward balance
from the time the GPU spent waiting for the CPU (kernel timestamps in the doorbell block): more than 30 us per layer raises
it, less than 8 us lowers it. `pcie_adapt=0` turns this off.

This machine: CPU 33.2 GB/s (the ggml kernel on one small matrix, optimistic against the 20 GB/s the real kernel gets),
PCIe 24.6 GB/s -> seed 0.43, 3 experts. 320 tokens, `-ub` 128, short prompt:

| Share | steps 33..320 step time | generation |
|---|---|---|
| adaptive from 0.43 (settles 0.43-0.49) | 40.5-43.9 ms | 21.9 t/s |
| fixed 0.43 | 39.3-43.3 ms | 22.2 t/s |
| fixed 0.55 | 40.5-45.7 ms | 22.3 t/s |

The optimum is flat between 0.43 and 0.55 here, so adaptation changes nothing on this machine and the calibrated seed is
already good. The value is on machines where the ratio differs (faster link, slower CPU) and at long context where the GPU
chain grows.

## One decode step under Nsight Systems (2026-10-12)

`nsys profile --trace=cuda --cuda-graph-trace=node`, short prompt, `-ub` 128, `pcie` on, calibrated plan; analysed
with `devspace\claude\hpcache-bench\step_analyze.py`. Profiling adds about 15% (layer period 930 us against 790 us).
One layer, microseconds from its `publish` kernel:

| From | To | What |
|---|---|---|
| 0 | 107 | publish, remap ids and weights, the pool experts (gate/up 53 us, down 29 us) |
| 107 | 402 | `pcie_stage_kernel` (295 us); the CPU serves its share in parallel |
| 402 | 455 | select, staged experts (21 + 11 us), `wait_merge` (11 us: the CPU was already done) |
| 455 | 826 | the rest of the layer on the GPU: attention or delta-net, shared expert, norms, router (370 us) |

Per decode step (GPU time): `pcie_stage_kernel` 15.1 ms (313 us median per layer), `mul_mat_vec_q` 11.9 ms,
`mul_mat_vec_f` bf16 6.5 ms, `wait_merge_kernel` 3.2 ms (mean 66 us, median 11 us), all other kernels about 6 ms.

Reading it: the stream is busy the whole step and the CPU is hidden (the GPU waits 3 ms per step for it). The stage
kernel moves about 3.5 MiB per layer in 295 us, 12 GB/s, half of the 24.6 GB/s the link gives to an isolated copy.
Probably the CPU kernels (20-23 GB/s) and the PCIe reads share the DRAM, which gives 40 GB/s read at best, so the
combined expert traffic is memory bound, not link bound. Not yet verified: run with the CPU idle (`pcie_frac=0.9`,
`pcie_adapt=0`) and compare the stage kernel's time per byte.

## Short-row bf16 mat-vec (core `ggml-cuda/mmvf.cu`, 2026-10-12)

The `up` hyper-connection projections (bf16, 320 columns x 10240 rows) ran as 10240 blocks of 160 threads at 35.4 us
(185 GB/s) against 18.0 us for the 10240-column `down` projection. New kernel `mul_mat_vec_f_short`: one warp per row,
four rows per block, shuffle reduction, used for f32/bf16, `n == 1`, no fusion, no ids, `ncols <= 512`, `nrows >= 2048`
(`GGML_CUDA_MMVF_NO_SHORT=1` selects the old kernel). 15 new `MUL_MAT` cases in `tests/test-backend-ops.cpp`
(m 2048/2050/10240, k 130/320/512, batch and broadcast dims): 1319 of 1319 tests pass on CUDA0.

Short prompt, `-ub` 128, 192 tokens, three alternating rounds:

| Kernel | step | generation |
|---|---|---|
| old | 42.3 / 42.1 / 42.1 ms | 21.5 / 21.3 / 21.4 t/s |
| new | 41.2 / 41.2 / 41.1 ms | 22.1 / 21.9 / 22.0 t/s |

About 0.9 ms per step (+2.8% decode), about half of the 1.7 ms the kernel's launch shape suggested: the new kernel
takes roughly 26 us per call, not the 18 us of its sibling.

## Where the stage kernel's time goes (2026-10-12, `devspace\claude\stage-bench`)

The provider's stage kernel (g experts of 3 tensors, 1.71 MiB per expert, read from a locked, mapped region) run in
isolation, 200 iterations with different experts each time, three kernel families and several grid shapes, with 0, 4
and 6 CPU threads streaming from DRAM at the same time. Time per call (us) and rate (GB/s):

| CPU threads streaming | g=1 | g=2 | g=3 | g=4 | rate |
|---|---|---|---|---|---|
| 0 | 82-87 | 155-175 | 226-253 | 307-353 | 20.3-23.8 GB/s |
| 4 | 112-134 | 233-253 | 349-365 | 467-521 | 13.4-15.5 GB/s |
| 6 | 127-141 | 242-263 | 363-401 | 494-518 | 12.7-14.8 GB/s |

All kernel variants (production loop, 4 and 8 loads in flight per thread, one block per tensor, z = 8/32/128 blocks)
land within 3% of each other at every row. A single `cudaMemcpyAsync` from the same region: 24.6 GB/s.

So the kernel is at the link's rate when the CPU is idle, and loses a third as soon as the CPU streams from the same
DRAM. That is also the 295-313 us seen in the Nsight timeline (g about 2.6, CPU busy). The CPU kernels (20-25 GB/s) and
the GPU's PCIe reads (14.5 GB/s under contention) add up to about the 40 GB/s the DRAM gives (membw: 8 threads, 41 GB/s).
Expert fetch is memory-bandwidth bound across both paths; the stage kernel cannot be made faster by changing its shape.
What moves the step is fewer bytes from DRAM per token: a higher VRAM hit rate (more or better placed slots), or smaller
experts.

## RAM upgrade: 64 GB at 3200 to 96 GB at 2400 (2026-10-12)

Four DIMMs (16 + 32 GB per channel, mixed makers), all running at 2400 MHz. `membw` read bandwidth by threads:

| Threads | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| 64 GB, 3200 MHz (GB/s) | 22.3 | 30.6 | 35.5 | 36.7 | 39.8 | 41.1 |
| 96 GB, 2400 MHz (GB/s) | 26.2 | 32.2 | 34.4 | 33.9 | 32.1 | 31.9 |

Peak 41.1 to 34.4 GB/s (-16%), at the provider's 6 threads 39.8 to 32.1 (-19%). Decode, calibrated plan, short prompt,
`-ub` 128, 192 tokens, hit rate unchanged (43-44%):

| | step | CPU experts | generation |
|---|---|---|---|
| 64 GB, 3200 MHz | 38.1-41.2 ms | 14-17 ms | 21.9-22.3 t/s |
| 96 GB, 2400 MHz | 44.6-45.7 ms | 18.7-19.6 ms | 20.2-20.6 t/s |

-7% decode for a bandwidth-bound design. The arena (40 GiB) and the PLE table (28 GiB) fit in 64 GB already, so the
extra capacity does not change what the provider can do.

### Same machine at 3200 MHz with 96 GB (2026-10-12, BIOS: manual 3200, 1.35 V, XMP off)

All four DIMMs report 3200. `membw` (three runs, GB/s): 1 thread 27.8-29.4, 2 threads 42.4-43.4, 4 threads 43.3-44.6,
6 threads 42.6-44.1, 8 threads 43.8-44.1. That is above the 64 GB / 3200 figures (6 threads 39.8): two ranks per channel.
(One run right after the setting showed 24.8 at 6 threads, a one-off; not reproduced.)

Decode, same code and settings as the "old kernel" rows of the short-row mat-vec test (192 tokens, `-ub` 128, hit 43.3%):

| | step | CPU experts | generation |
|---|---|---|---|
| 64 GB, 3200 MHz (like for like) | 42.1-42.3 ms | 16.4 ms | 21.3-21.5 t/s |
| 96 GB, 2400 MHz | 44.6-45.7 ms | 18.7-19.6 ms | 20.2-20.6 t/s |
| 96 GB, 3200 MHz | 42.2-42.4 ms | 16.6-16.8 ms | 21.3-21.4 t/s |

The 21.9-22.3 t/s given for the 64 GB row in the table above came from 320-token windows with a 50% hit rate, not from
the same run; against the like-for-like row the 2400 MHz loss was -5.5% and 3200 MHz gives it all back. The step did not
get faster although `membw` reads 10% more: the decode is not limited by the raw read rate alone.

## KV cache type against decode at 16K tokens of context (2026-10-12)

`kvsweep.ps1`: 16K-token prompt, `-c 32768`, `-ub` 512, 96 generated tokens, calibrated seed, first 32-step windows
after the prefill (the context is about 16K), 96 GB at 3200 MHz. The K and V types are the same.

| KV type | pool slots | hit | CPU experts | GPU side (step - CPU) | step | prefill | decode |
|---|---|---|---|---|---|---|---|
| q5_0 | 979 | 33.9% | 20.3 ms | 33.2 ms | 53.4 ms | 239.6 t/s | 18.4 t/s |
| q8_0 | 878 | 32.6% | 21.0 ms | 30.9 ms | 51.9 ms | 239.6 t/s | 18.7 t/s |
| q4_0 | 1012 | 34.7% | 19.9 ms | 31.3 ms | 51.1 ms | 240.2 t/s | 19.1 t/s |
| **f16** | 626 | 28.9% | 21.3 ms | **26.9 ms** | 48.2 ms | 240.5 t/s | **20.3 t/s** |

The quantized types cost 4-6 ms of GPU time per step at 16K tokens (dequantizing the whole cache for each token),
which is the long-context slope of 0.36-0.39 ms per 1000 tokens measured earlier. f16 has no dequantization, so it wins
by 10% even with 36% fewer pool slots and 5 points less hit rate. The dequantization cost grows with the context
used; the slots lost grow with the context allocated (`-c`). The crossover with q5_0 is at about 5-6K tokens used.

### f16 against q5_0 with the context filled (2026-10-12, `kvfull.ps1`, `kvfull_long.ps1`)

`-c 36864`, calibrated plan, 96 GB at 3200 MHz, 128 generated tokens, K and V of the same type.

| Context used | KV | pool slots | hit | step | CPU experts | prefill | decode |
|---|---|---|---|---|---|---|---|
| about 1K (short prompt, `-ub` 128) | q5_0 | 1300 | 36.9% | 45.6 ms | 19.2 ms | | 20.8 t/s |
| about 1K | f16 | 904 | 32.2% | 45.7 ms | 20.8 ms | | 21.1 t/s |
| about 32K (100,000 characters, `-ub` 512) | q5_0 | 955 | 36.7% | 57.8 ms | 20.8 ms | 231 t/s | 16.4 t/s |
| about 32K | f16 | 559 | 27.1% | 48.0 ms | 21.3 ms | 232 t/s | **19.6 t/s** (+20%) |

(16K used, `-c 32768`: f16 +10%, see the sweep above.) f16 ties at a short context and wins more the longer the
context; the dequantization of the quantized types grows with the tokens used, while f16 only gives up pool slots.

## Batched table uploads and the short-row kernel together (2026-10-12)

Table uploads: one device array for the slot tables of all layers with a pinned mirror, one `cudaMemcpy` per round
(`hpcache_table_block`, `stage_table`, `flush_tables`). `admission per step` (us), before and after:

| | publish | select | copy issue | table upload |
|---|---|---|---|---|
| per-layer uploads | 215-485 | 41-51 | 2-3 | 178-422 |
| one upload per round | 66-74 | 42-45 | 2-3 | 36-54 |

The admission runs after the step timer stops, so it shows in the generated tokens per second, not in the step column.
Short prompt, `-ub` 128, 192 tokens, three alternating rounds, 96 GB at 3200 MHz (before: 21.3-21.5 t/s):

| | step | generation |
|---|---|---|
| batched uploads, old mat-vec kernel | 42.3-42.4 ms | 21.6 / 21.8 / 21.7 t/s |
| batched uploads, short-row kernel | 41.2 / 41.3 / 41.9 ms | 22.4 / 22.4 / 21.9 t/s |

About +0.35 t/s from the uploads (+1.6%) and +0.5 from the kernel (+2.3%). `MUL_MAT` backend tests: 1319 of 1319.

## Correction: Strata's hit rate is not per token (2026-10-12)

The "about 50% hit, 82 us per missed expert" comparison in "Like-for-like with Strata no-drafts" is not like for like.
Strata's native pack needs a verify window of at least 2 tokens (`--spec 1` is refused: "needs --spec T (T >= 2)"), and
with no drafter the second position is a constant token 0 (`generate.cpp:4874-4875`: `d = at < oracle.size() ? oracle[at]
: 0`). Its tier statistics (CPU 4.64 distinct / 5.01 routed, PCIe 4.59 distinct per layer, reproduced here at 21.88 t/s)
cover both positions, and its usage counters feed on both, so the adaptive tier keeps the experts of token 0 resident.
Per generated token Strata fetches about 9.2 distinct missed experts per layer, this provider about 5.7. The real
token's hit rate in Strata is not observable from its output; if the constant token hits at 60-70% it is 30-40%, below
the 43% here. The no-draft speeds (21.7-21.9 t/s) are comparable; the hit rates are not.
