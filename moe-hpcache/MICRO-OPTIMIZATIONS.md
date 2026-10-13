# Micro-optimizations to revisit

Candidates found while reading the provider. None is on the critical path of a decode step except where marked; each
has the place, the measured or estimated cost, and what would change. Do these after the larger items (M2 batched
decode, long-session validation) and measure before and after: the step is about 38-45 ms and a 1% change is below the
noise of a single run (run each configuration three times, see `BASELINES.md`).

## Worth doing (measurable)

### 1. Table uploads: one copy per round instead of one per layer
- Where: `hpcache_decoder::publish_pending`, `hpcache_decoder::admit`, `hpcache_cuda_table_upload` (`hpcache-cuda.cu`).
- Cost: `admission per step` line of the arena run: publish 485 us + table upload 422 us, about 0.96 ms of a 59 ms
  step (1.6%). Almost all of it is a synchronous `cudaMemcpy` of one 2 KiB table per touched layer (about 20 layers per
  round, uploaded twice per round: once when new experts are mapped, once after the victims are unmapped).
- Change: one device array for all layer tables (48 x 512 x 4 B = 96 KiB), a pinned host copy, and one async copy per
  round on the copier stream; map new experts and unmap victims in the same upload.
- Estimate: about 0.9 ms per step, 2% decode (about +0.4 t/s).
- Measure: `HPCACHE_STATS_EVERY` windows; the `admission per step` line at exit.

### 2. Arena load: reach the drive's rate
- Where: `hpcache_arena::open`, step 4 (read) in `hpcache-arena.cpp`.
- Cost: 40.01 GiB in 14.5 s (2.96 GB/s); the drive does 3.46 GB/s (`nvme-load`), so 12 s is the floor.
- Change: queue depth 8 instead of 4, and start the second shard's requests while the first drains (today each shard
  is read by a separate `read_jobs` call that waits for its tail). Padding to 2 MiB per layer costs about 96 MiB of
  reads and could be dropped for unlocked layers.
- Estimate: about 2 s per start.

### 3. Linux loader depth
- Where: `read_jobs` POSIX branch in `hpcache-arena.cpp` (one `pread` at a time).
- Cost: depth 1 with 2 MiB requests is 2.34 GB/s against 3.46 GB/s at depth 2 or more (measured on this NVMe).
- Change: 4 worker threads issuing `pread` with `O_DIRECT`, or io_uring when present. Not built or run here (Windows
  only); keep it behind the same function.

## Cheap cleanups (not measurable, readability)

### 4. Arena layout: `stable_sort` and the second container
- Where: step 2 of `hpcache_arena::open`.
- `stable_sort` on a unique key (layer, name) gives the same result as `sort`. The `tensors` vector is a build-time copy
  of what `index` (an `unordered_map` by name) keeps afterwards; one `std::map` keyed by `(layer, name)` would serve both
  (iteration for the layout, lookup for `find`). Note the key must hold the layer: a map by name alone orders `blk.10`
  before `blk.2`.
- Cost: microseconds over 144 tensors, once, out of a 14.5 s `open()`. Left as it is by decision.

### 5. `touched` sort and unique in `publish_pending` and `admit`
- At most 24 pointers per round; a 64-bit layer mask removes the sort and the erase. Goes away with item 1.

### 6. Condition variable in `copier_wait`
- It returns at once when the copies are done (they were queued a step earlier), and blocks only when they are not.
  An atomic flag with `std::atomic::wait` is lighter by a few hundred nanoseconds per step. Not worth a change alone.
- Measured (the `copier:` line at exit; 192 tokens, short prompt):

  | Case | batch landed (mean / max) | next round waited | found still running |
  |---|---|---|---|
  | default (24 swaps, arena locked) | 1.9 ms / 3.1 ms | 0.1 us | 0 of 187 |
  | 96 swaps per round | 2.7 ms / 29.6 ms (first fill) | 0.2 us | 0 of 187 |
  | 400 swaps per round | 2.7 ms / 28.4 ms (first fill) | 0.2 us | 0 of 187 |
  | arena not locked (pageable source) | 5.0 ms / 7.8 ms | 0.1 us | 0 of 187 |

  The copies land in about 2 ms against a 40 ms step, 20 times the margin, and the wait never blocked, even with the
  source unlocked. Nothing to fix here. The `publish` line of `admission per step` (215-345 us) is the table uploads
  (item 1), not this wait.

## Known costs of the GPU path (design, not bugs)

### 7. Fixed cost of the path when nothing is staged
- `pcie=1,pcie_frac=0` measured 43.1 ms against 42.4 ms (+0.7 ms): the select, stage and extra graph ops run for every
  layer even when `g = 0`. A layer-level skip is not possible without rebuilding the graph; reducing it needs fewer ops
  (fuse the select kernel into the stage kernel, write ids and weights in one launch).

### 8. Padding slots in the staged compute
- The staged expert graph always computes `pcie_max` slots; unused ones are zeroed and carry weight 0. With
  `pcie_max=3` and an average of two staged experts per layer, one slot's compute is wasted per layer.

## Measurements to refresh

- The calibrated CPU rate (33 GB/s) comes from the ggml kernel on one small matrix and is optimistic against the
  20-23 GB/s the real kernel reaches. The run-time adjustment hides it; a calibration that times the provider's own
  kernel on a layer-sized set would seed the share exactly.
- VNNI width: the comment in `hpcache-cpu.cpp` says the 512-bit variant lost because of a clock drop, which was never
  measured. A micro-benchmark under `devspace\claude\vnni-width` (both widths, weights in L1, L2, DRAM) would settle it.
