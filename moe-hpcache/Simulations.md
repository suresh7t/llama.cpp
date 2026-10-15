# Simulations

Offline studies of the VRAM expert pool, run on recorded routing traces. They answer what a policy or a bigger pool can
give before any provider code is written. Nothing here runs on the GPU. The scripts live outside the repository, in
`C:\Users\suresh\devspace\claude\hpcache-bench` (`sim5.py`, built on `sim4.py`).

## Hit-rate bounds for the expert pool (2026-10-15)

### Question

How much can the pool hit rate still improve through policy (which experts are resident), and how much through size
(slots)? The hit rate is the share of routed experts of a decode step that are resident in VRAM. Every miss is read from
host memory, by the CPU or by the GPU over PCIe, and host memory bandwidth is what limits the step.

### Method

- **Traces:** routed expert ids per token and layer, recorded in `serve()` with `HPCACHE_ROUTING_TRACE`. One record is
  `il, id[0..9]` as int32, 48 layers per token. Single-token decode only (the prefill path is not recorded).

  | Trace | Tokens | Prompt |
  |---|---|---|
  | story | 299 | short story (lighthouse keeper) |
  | lru | 399 | long multi-turn code task (LRU cache in C) |
  | code | 499 | C++ LRU cache with a hash map and a linked list, plus a test |
  | chat | 499 | explain TCP congestion control, compare Reno with CUBIC |
  | essay | 499 | essay on the causes and consequences of the French Revolution |

- **Pool seeding:** the first N entries of the Strata `expert-profile.bin` ranking, as in the provider. Capacity per
  layer is fixed at the count of profile entries of that layer, unless a variant says otherwise.
- **Live policy:** a copy of `admit()`: scores incremented per access, decayed by 0.9 per step, one candidate per layer,
  victim is the coldest resident of the same layer, up to 24 swaps per step globally ranked by gain, threshold 2.0,
  margin 1.0, copies land one step late (`async_swaps=1`).
- **Variants:**
  - `shared capacity`: the victim is the coldest resident of any layer, so slots move between layers.
  - `static, hindsight`: the best fixed set of N (layer, expert) pairs for that trace, counted on the trace itself.
  - `static, other traces`: the best fixed set counted on the other four traces, tested on this one.
  - `warm start`: the live policy started from the final state (residents and scores) of another trace.
  - `Belady`: optimal replacement over the whole pool (MIN with bypass), swaps are free and instant. Upper bound for any
    online policy at that pool size.
- **Validation:** the simulation of the live policy at 1,300 slots gives 49.9% (mean of five traces); the real runs of
  2026-10-14 gave 48 to 53% cumulative and 50 to 57% in the third 100-step window.

### Result at 1,300 slots (mean of five traces, all tokens)

| Policy | Hit rate | Per trace: story / lru / code / chat / essay |
|---|---|---|
| Strata profile only, no admission | 15.0% | 7.9 / 23.6 / 20.9 / 14.2 / 8.4 |
| **Live policy (as built)** | **49.9%** | n/a |
| Live policy, 12 swaps per step | 48.0% | n/a |
| Live policy, 48 swaps per step | 50.5% | n/a |
| Shared capacity across layers | 51.7% | n/a |
| Shared capacity, 48 swaps per step | 52.4% | n/a |
| Static best set, hindsight | 51.6% | 48.9 / 54.5 / 50.8 / 47.6 / 56.2 |
| Static set from the other traces | 19.3% | 8.6 / 35.2 / 27.9 / 15.1 / 9.8 |
| Live policy, warm start from another prompt | 49.0% | n/a |
| Belady optimal (upper bound) | 74.3% | 75.6 / 73.1 / 73.4 / 71.9 / 77.7 |

Belady needs 78 insertions per token at 1,300 slots (about 130 MiB per token at 1.67 MiB per expert), against about 20
for the live policy.

### Result by pool size (live policy, mean of five traces)

| Slots | Share of the 24,576 expert-layers | Live | Shared capacity | Static, hindsight | Belady |
|---|---|---|---|---|---|
| 878 | 3.6% | 42.7% | 44.1% | 42.9% | 67.3% |
| 1,300 | 5.3% | 49.9% | 51.7% | 51.6% | 74.3% |
| 1,619 | 6.6% | 54.1% | 56.1% | 56.9% | 78.1% |
| 2,000 | 8.1% | 58.5% | 60.2% | 62.2% | 81.6% |
| 2,600 | 10.6% | 64.0% | 65.3% | 69.0% | 85.4% |
| 3,600 | 14.6% | 70.5% | 71.3% | 77.7% | 89.6% |

### Conclusions

1. **The live policy is at the best fixed set for each prompt** (49.9% against 51.6% with hindsight at 1,300 slots).
   Its dynamics, not the seed, produce the hit rate: the Strata profile alone gives 15%.
2. **A workload-matched or persisted profile does not help.** The hot set is specific to the prompt: a set built from
   the other traces gives 19%, and starting from another prompt's final state is slightly worse (-0.9 points).
3. **The policy has under 2 points left.** Sharing capacity across layers gives +1.8 points (about 0.15 t/s) and needs
   a shared slot array with a per-layer table instead of per-layer tensors with fixed slot counts. More swaps per step
   give +0.6 from 24 to 48 and nothing beyond 48 (the live measurements in `BASELINES.md` found the plateau at 24 to 36).
4. **The Belady bound is not realizable.** It needs knowledge of the future and 3 to 4 times the swap traffic.
5. **Slots are the lever.** About +1.3 points per 100 slots at 1,300 to 1,600 slots, falling to about +0.9 per 100
   further out. 2,600 slots would be about +14 points.
6. **Hit rate to speed:** +100 slots was measured at about +0.1 t/s (16K and 8K context runs of 2026-10-14), so +10
   points of hit rate is about +0.8 to 1 t/s. This is an estimate from those runs and the balanced CPU/PCIe split, not a
   direct measurement.

### Limits

- Traces are 300 to 500 tokens of single-token decode, one sequence. Long sessions may converge differently.
- The simulation counts the ten experts of a layer as simultaneous accesses of the state before the step, and models the
  one-step copy delay; it does not model copy bandwidth or the time a swap takes on the copier thread.
- Belady is optimal for the hit count only; it ignores that CPU and PCIe misses have different costs.
- Five prompts, one model (Qwen3.8 Flash Next IQ3_XXS) and one expert profile.

### Reproduce

```
cd C:\Users\suresh\devspace\claude\hpcache-bench
python sim5.py 878 1300 1619 2000 2600 3600
```

New traces: run `llama-cli` with the provider and `HPCACHE_ROUTING_TRACE=<file>`, greedy, `--ignore-eos`, 500 tokens
(the files above are `trace-story.bin`, `trace-lru.bin`, `trace-code.bin`, `trace-chat.bin`, `trace-essay.bin`).
