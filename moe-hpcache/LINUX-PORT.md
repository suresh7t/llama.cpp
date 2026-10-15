# moe-hpcache: Linux port verification steps

Audience: an agent (or person) on Linux / WSL-Ubuntu with the repo checked out on the `moe-hpcache` branch.
Goal: confirm the provider builds, loads and decodes correctly on Linux, and list what is untested.
Rules: do not edit core files outside `moe-hpcache/` and `common/moe-provider.*` without asking; ASCII only;
do not commit; put scratch scripts under a folder outside the repo; report numbers, do not guess.

## Scope (owner decision)

Success = it compiles and links cleanly on Linux and the provider loads and decodes. Slow I/O (O_DIRECT over
9p, single-request pread) and a failed or small pinned arena on WSL are acceptable. Sections 4 and the
performance numbers in section 5 are optional; do not spend effort on speed.

## 0. Known state (read first)

- The code was only built and run on Windows 11 (MSVC, CUDA 13, RTX 3060 Ti). Linux paths exist but were
  never compiled or run. Treat every `#else` branch of `_WIN32` as untested.
- Files with platform branches (grep `_WIN32`):
  - `hpcache-arena.cpp`: Windows uses VirtualAlloc (2 MiB aligned) + IOCP overlapped unbuffered reads, depth 4.
    Linux uses `open(O_DIRECT)`, `mmap(MAP_ANONYMOUS)` and plain `pread`, ONE request at a time (slow, weak spot).
  - `hpcache-io.cpp`: `O_DIRECT` + `posix_memalign` for the lazy expert loader.
  - `moe-provider.cpp` (line ~17, ~29, ~200): library export macro, process / thread pinning (`pin`, `cpus=`).
  - `calibrate/main.cpp`: host-arena probe (`mmap`, cudart `cudaHostRegister`).
  - `common/moe-provider.cpp`: provider library loading (`--moe-provider hpcache` -> `libmoe-hpcache.so`?
    check the name the loader builds; on Windows it is `moe-hpcache.dll`).
- WSL2 caveats: `O_DIRECT` on the Windows drives (`/mnt/c`, `/mnt/d`) over 9p is slow or unsupported; copy the
  GGUF into the WSL ext4 disk first. `cudaHostRegister` of tens of GiB is limited by the WSL memory cap
  (`.wslconfig` memory=) and by WSL's pinned-memory support; a failure there is expected, not a bug. The
  decisive Linux check is a real Linux host if one exists; WSL results are a smoke test only.

## 1. Build

1. Configure with CUDA, Release, a real arch only: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
   -DCMAKE_BUILD_TYPE=Release` into `build-linux/` (do not reuse the Windows `dbgbld-vs`).
2. Build targets: `llama-server llama-cli llama-bench` and the provider target (see `moe-hpcache/CMakeLists.txt`)
   and the calibrate tool.
3. Record every compile error. Expected suspects: missing `#include <unistd.h>/<fcntl.h>/<sys/mman.h>`,
   `ssize_t`, `std::filesystem`, MSVC-only intrinsics or pragmas, `__declspec(dllexport)` without a GCC
   `__attribute__((visibility("default")))` branch, `.dll` string literals, backslash path separators, `_aligned_malloc`.
   Fix in place with `#if defined(_WIN32)` / `#else` branches, keep Windows behaviour byte-identical.
4. Confirm the CPU kernel picks the right ISA: build with `-march=native` or the project's flags and check that
   the AVX2 and AVX512-VNNI paths in `hpcache-cpu.cpp` compile under GCC (inline asm / `_mm512_dpbusd_epi32`
   needs `-mavx512vnni`). Run once with `HPCACHE_NO_VNNI=1` and once without; both must give the same output.

## 2. Load and smoke test

Model: the same IQ3_XXS GGUF (copy to ext4). Use `--device CUDA0` explicitly (rule of this project).

1. Stock run, no provider, `--cpu-moe`: record t/s as the Linux baseline.
2. `--moe-provider hpcache` with no plan: must load, print the provider banner, decode.
3. With `--moe-provider-args "arena=1"`: check the log line that reports the arena size, pinned bytes and the
   "first tensor verified" message. Then `arena=0` as the fallback path.
4. Greedy determinism: `--temp 0 -n 64` same prompt, provider vs stock; text should match for the first
   tokens, small divergence later is the known ~0.018 KLD path noise. Use `HPCACHE_VERIFY=1` to compare the
   provider's expert output against the ggml reference per layer; it must pass.
5. Run `llama-server` with `--moe-provider hpcache`, send one chat request, check the response is coherent.

## 3. Calibrate

1. Run the calibrate tool the way the README says; check the host_arena probe prints a sane maximum (on WSL it
   may be small or fail; record the value and the error text) and that the written plan contains
   `host_arena` and `expert_split`.
2. Feed that plan back with `plan=...` and confirm the log reports the seeded GPU share.

## 4. Linux-specific work items (only after sections 1-3 pass)

Do these in order, one measurement each, report before moving on:

1. Arena reader depth: replace the single-request `pread` loop in `hpcache-arena.cpp` with a worker pool
   (4-8 threads, 2 MiB O_DIRECT reads, into the 2 MiB aligned mmap region) or io_uring when available. Measure
   load time of the 40 GiB expert set; Windows reference is about 3.4 GB/s on this NVMe (device limit).
2. Huge pages: try `madvise(MADV_HUGEPAGE)` on the arena before `cudaHostRegister`; keep only if load or
   decode improves.
3. Pinning: replace the Windows affinity code with `sched_setaffinity` on the physical cores (skip SMT
   siblings, read `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`). Check `-t 6` throughput.
4. `RLIMIT_MEMLOCK`: `cudaHostRegister` needs a high limit (`ulimit -l unlimited` or limits.conf); make the
   failure message say so.
5. NUMA: single node assumed; log if more than one node is present.

## 5. Acceptance report (what to send back)

A table with: build result per target, each test in section 2 (pass/fail + text), calibrate output,
t/s decode and prefill for stock vs provider on Linux, arena load time, any code changes (file + one line
each), and the list of things that could not be tested (e.g. WSL pinned-memory limits). Windows reference
numbers for comparison: decode 21.6-21.8 t/s at short context, prefill 230-240 t/s at `-ub 512`, both with
KV f16, `-t 6`, 96 GB RAM.
