# moe-hpcache — Heterogeneous Parallel (HP) Execution

## Build and module boundary

HP-cache implementation code is built as the standalone shared library
`moe-hpcache.dll` on Windows and `moe-hpcache.so` on Unix-like systems. The
calibration executable is built under the same module directory. The llama
core and common layers contain only wiring, argument handling, and the generic
callback boundary. Placement, scheduling, direct I/O, lazy maps, and plan
parsing remain replaceable inside the shared library so they can evolve without
rebuilding the llama core.

## 1. Purpose

`moe-hpcache` extends the MoE execution model in `llama.cpp` with **Heterogeneous Parallel (HP) execution**.

The goal is to allow a single MoE operation to make use of **all computational backends available on the system**:

* CPU
* CUDA
* Vulkan
* potentially other ggml backends in the future

The important architectural constraint is that HP should **work within the existing `llama.cpp` / ggml backend architecture**, rather than creating a new execution backend that duplicates CPU, CUDA, or Vulkan functionality.

The scheduler owns **policy**.

The ggml backends own **execution**.

The page cache owns **residency**.

The GGUF/NVMe layer owns **persistent storage**.

---

# 2. Architectural Boundary

The intended architecture is:

```text
                         llama.cpp
                            │
                   ┌────────┴────────┐
                   │                 │
             MoE Scheduler          ggml
                   │                 │
                   │       ┌─────────┼─────────┐
                   │       │         │         │
                   │      CPU       CUDA     Vulkan
                   │
                   └── expert placement/cache

             Scheduler owns policy.
             ggml backends own execution.
             Cache owns residency.
             GGUF/NVMe layer owns storage.
```

This separation is fundamental.

HP does **not** attempt to make CPU, CUDA, and Vulkan implementations understand each other.

Instead, the scheduler determines:

1. which experts/tensor portions are required;
2. where those data are resident;
3. which backend should execute each portion;
4. how work should be partitioned;
5. when independent backend operations can execute concurrently;
6. how the resulting partial tensors are combined.

The existing ggml backend implementations remain responsible for performing the actual tensor operations.

---

# 3. Relationship to llama.cpp Discussion #14514

Discussion **#14514** is the primary architectural reference for HP execution.

The relevant idea is the evolution of ggml's graph execution/planning model toward allowing execution to be planned around backend capabilities and, importantly, allowing independent work to overlap rather than treating execution as a strictly serialized sequence.

HP should build on this direction rather than introduce a parallel execution framework outside ggml.

The central question for `moe-hpcache` is therefore:

> How can the MoE scheduler express heterogeneous parallel work using the existing ggml graph/backend execution machinery?

This is preferable to creating a new abstraction such as:

```text
ParallelBackend
```

or:

```text
HPBackend
```

because CPU, CUDA, and Vulkan are already legitimate ggml backends.

HP is an **execution policy**, not another backend.

---

# 4. HP Execution Model

For an MoE operation, the router identifies the experts required for the current token(s).

Instead of assigning all required work to one backend:

```text
             MoE operation
                   │
                   ▼
                 CUDA
                   │
                   ▼
                result
```

HP allows the scheduler to partition the work:

```text
                    MoE operation
                          │
                  required experts
                          │
                 HP scheduling policy
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
          CPU           CUDA         Vulkan
            │             │             │
            ▼             ▼             ▼
        partial R0     partial R1     partial R2
            │             │             │
            └─────────────┼─────────────┘
                          │
                       merge
                          │
                          ▼
                       result
```

The three operations can execute independently when their dependencies and backend execution mechanisms permit it.

The scheduler therefore creates **parallel execution opportunities**; ggml remains responsible for executing the operations.

---

# 5. HP Is Not Traditional Multi-GPU Tensor Parallelism

The HP model is broader than conventional tensor parallelism.

Traditional tensor parallelism generally assumes multiple similar accelerators:

```text
GPU 0 ─┐
       ├── tensor parallel computation
GPU 1 ─┘
```

HP allows heterogeneous resources:

```text
CPU ─────┐
CUDA ────┼── heterogeneous parallel computation
Vulkan ──┘
```

The resources do not need to have identical performance, memory capacity, or architecture.

The scheduler may therefore make asymmetric decisions.

For example:

```text
CUDA:
    large/high-bandwidth resident portion

Vulkan:
    smaller resident portion

CPU:
    overflow / cache miss / lower-priority portion
```

The exact partitioning policy is an implementation concern of the scheduler.

---

# 6. Relationship to #24528

Discussion **#24528** provides the complementary foundation for **MoE expert residency and caching**.

The cache answers:

> Where is the expert data?

HP answers:

> Where should the computation execute?

These are related but separate questions.

```text
                  Expert request
                       │
                       ▼
                moe-pagecache
                       │
             ┌─────────┼─────────┐
             │         │         │
            CPU       CUDA     Vulkan
           resident  resident  resident
             │         │         │
             └─────────┼─────────┘
                       │
                       ▼
                  HP Scheduler
                       │
             dispatch according
               to residency
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        CPU          CUDA         Vulkan
          │            │            │
          └────────────┼────────────┘
                       ▼
                      merge
```

Thus:

**#24528 → residency/cache**

**#14514 → graph/backend execution architecture**

**`moe-hpcache` → integrates the two through MoE scheduling**

---

# 7. Ownership Model

## 7.1 MoE Scheduler

Owns:

* expert routing interpretation;
* expert placement policy;
* backend selection;
* partitioning of work;
* deciding which backend receives which tensor/expert work;
* identifying independent operations;
* dispatching parallel work;
* coordinating completion;
* initiating result combination.

The scheduler should not contain CUDA, Vulkan, or CPU kernel implementations.

---

## 7.2 ggml Backend

Owns:

* tensor kernels;
* backend-specific memory;
* backend-specific streams/queues;
* synchronization required by the backend;
* execution of the operations assigned to it.

Examples:

```text
ggml-cuda
ggml-vulkan
ggml-cpu
```

HP should use these existing implementations.

---

## 7.3 moe-pagecache

The implementation uses `moe-hpcache/hpcache-lazy-map.h` for both the PLE and expert
resident-value maps. Every known key keeps its source metadata after its value
is evicted. Each map has its own byte budget. The hash table is sharded into
sixteen buckets: hits lock only the bucket that owns the key, while miss loading
and eviction use one mutation lock to keep the byte budget exact.
The standard C++ library has no atomic hash map, so this avoids an external
concurrent-container dependency while reducing contention on decode hits.

Owns:

* page residency;
* page allocation;
* cache lookup;
* LRU/replacement policy;
* CPU/GPU cache state;
* page movement;
* host-pinned memory;
* asynchronous storage reads;
* mapping logical expert data to resident pages.

The cache should not decide which backend performs the computation.

### Lazy tensor load policy

HP-cache uses a generic tensor load policy callback:

* `LOAD_IMMEDIATE` allocates and loads the tensor during model load.
* `LOAD_DEFAULT` keeps the normal llama.cpp loading behavior.
* `LOAD_LAZY` records the GGUF source range and tries to make the value
  resident if it fits the cache budget. If the budget is full, only metadata is
  retained.

On a later decode miss, `LOAD_LAZY` materializes the value with Direct I/O,
evicts unpinned resident values when needed, and then executes the operation.
Dense weights and KV tensors use immediate loading. PLE and expert tensors use
the lazy policy. At debug verbosity, the callback logs every tensor name,
source offset, size, and selected policy.

---

## 7.4 GGUF / NVMe Layer

Owns:

* persistent GGUF representation;
* file offsets;
* tensor storage;
* NVMe/file I/O;
* loading pages that are not currently resident.

---

# 8. Execution vs Residency

An important design principle is:

> **Residency does not imply execution ownership.**

For example, an expert may be resident in CPU memory but still be selected for execution on CUDA.

That may require a transfer:

```text
CPU resident expert
        │
        ▼
   CUDA transfer
        │
        ▼
   CUDA execution
```

Similarly, a page may be resident on CUDA because it was recently used, while the scheduler may deliberately execute another portion on CPU.

Therefore:

```text
Cache state
     ≠
Execution decision
```

The scheduler combines cache state, backend capability, transfer cost, and workload characteristics to make the execution decision.

---

# 9. First HP Prototype

The first implementation should deliberately avoid solving the complete optimization problem.

The initial prototype should establish the execution mechanism.

### Phase 1

Given a tensor operation:

```text
T
```

partition it into independent portions:

```text
T0
T1
T2
```

and assign:

```text
T0 → CPU
T1 → CUDA
T2 → Vulkan
```

Execute the three operations independently.

Then combine:

```text
R = merge(R0, R1, R2)
```

The first prototype should measure:

* CPU execution time;
* CUDA execution time;
* Vulkan execution time;
* host/device transfer time;
* synchronization time;
* merge time;
* total wall-clock time.

The purpose is to validate that the **execution model works**, not to optimize scheduling yet.

---

# 10. Merge Semantics

The merge operation must be defined according to the tensor operation being parallelized.

The scheduler should not assume that every operation can simply concatenate results.

Possible cases include:

```text
concatenation
reduction
element-wise accumulation
matrix/tensor partial-result reduction
expert-output accumulation
```

For MoE, the final operation generally combines the contributions of the selected experts according to the router weights.

Therefore HP needs an explicit operation-level definition of:

```text
partial result
      ↓
combination operation
      ↓
final result
```

The implementation should use ggml operations wherever possible rather than introducing backend-specific merge code into the scheduler.

---

# 11. Why #14514 Matters

The major architectural question is not:

> "How do we write CPU + CUDA + Vulkan kernels?"

Those already exist.

The question is:

> "How do we represent and schedule independent operations so that existing ggml backends can execute them concurrently?"

This is where the execution/planning direction discussed in **#14514** becomes important.

The desired model is:

```text
              Graph / execution plan
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
        CPU task    CUDA task    Vulkan task
          │            │            │
          │            │            │
          └────────────┼────────────┘
                       │
                 dependency
                    barrier
                       │
                       ▼
                    merge
```

The graph/execution system should expose enough information for independent backend operations to overlap while respecting dependencies.

HP should therefore investigate how the existing ggml graph planner and backend scheduler can represent this pattern before adding new abstractions.

---

# 12. Initial Questions to Answer in llama.cpp

Before implementing HP, inspect the current implementation around:

1. `ggml_backend_t`
2. `ggml_backend_dev_t`
3. backend buffer interfaces
4. graph planning
5. graph execution
6. backend scheduling
7. synchronization/event mechanisms
8. CUDA streams
9. Vulkan queues
10. CPU execution
11. tensor splitting / multi-device execution
12. MoE expert dispatch
13. the implementation associated with #24528

The first engineering task is therefore **code archaeology**, not coding.

We need to determine exactly where the current architecture permits:

```text
backend A operation
        │
        ├────────── concurrent ──────────┐
        │                                │
backend B operation                      │
        │                                │
        └────────────────────────────────┘
                         │
                      barrier
                         │
                       merge
```

---

# 13. Proposed Development Sequence

### Step 1 — Understand #14514

Map the discussion's proposed/current graph-planning architecture to the current llama.cpp source tree.

Document:

```text
graph construction
      ↓
graph planning
      ↓
backend assignment
      ↓
graph execution
      ↓
synchronization
```

---

### Step 2 — Understand #24528

Map the MoE expert-cache implementation to:

```text
expert request
      ↓
cache lookup
      ↓
resident?
   ┌──┴──┐
  yes    no
   │      │
   │    storage
   │      │
   │    page load
   │      │
   └──┬───┘
      ▼
   available
```

---

### Step 3 — Define the HP scheduling interface

Do not implement an entire scheduler immediately.

Define the minimum information required:

```text
HP Work Item

tensor/expert:
    logical identity

data:
    pages required

candidate backends:
    CPU / CUDA / Vulkan

residency:
    CPU / CUDA / Vulkan / storage

partition:
    tensor range / expert range

dependencies:
    prerequisite operations

completion:
    backend completion event
```

---

### Step 4 — Implement a synthetic parallel operation

Before connecting the full MoE cache:

```text
input tensor
    │
    ├── CPU
    ├── CUDA
    └── Vulkan
         │
         ▼
       merge
```

This establishes the HP execution mechanism independently of the page cache.

---

### Step 5 — Connect `moe-pagecache`

Once the parallel execution mechanism works:

```text
moe-pagecache
      │
      ▼
expert/page residency
      │
      ▼
HP scheduler
      │
      ▼
backend assignment
      │
      ▼
parallel execution
      │
      ▼
merge
```

This gives `moe-hpcache` its complete behavior.

---

# 14. Design Principle

The project should preserve the following rule throughout implementation:

> **The scheduler owns policy; ggml backends own execution; the cache owns residency; GGUF/NVMe owns storage.**

HP adds one capability to the scheduler:

> **The scheduler may exploit independent computational capacity across multiple ggml backends simultaneously.**

It does not replace ggml.

It does not create a universal backend.

It does not require CPU, CUDA, and Vulkan to share implementation details.

It simply coordinates them.

---

# 15. Working Terminology

For this branch:

**HP**
: Heterogeneous Parallel execution.

**HP Scheduler**
: The policy layer deciding how work is partitioned and assigned across available ggml backends.

**moe-pagecache**
: The residency subsystem for expert/tensor pages.

**moe-hpcache**
: The integrated implementation combining MoE page residency with heterogeneous parallel execution.

**Backend**
: Existing ggml execution implementation such as CPU, CUDA, or Vulkan.

**Work Item**
: A schedulable portion of an MoE/tensor operation assigned to a backend.

**Partial Result**
: The output generated by one backend for its assigned work.

**Merge**
: The ggml operation that combines partial results into the final tensor result.

---

# 16. Architectural Target

The final architecture should remain close to:

```text
                         llama.cpp
                            │
                     ┌──────▼──────┐
                     │ MoE Router  │
                     └──────┬──────┘
                            │
                     required experts
                            │
                   ┌────────▼────────┐
                   │   HP Scheduler  │
                   │                 │
                   │ placement       │
                   │ residency       │
                   │ partitioning    │
                   │ backend choice  │
                   │ parallel launch │
                   │ synchronization │
                   └────────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
           ggml CPU      ggml CUDA    ggml Vulkan
              │             │             │
              └─────────────┼─────────────┘
                            │
                       partial results
                            │
                            ▼
                       ggml merge
                            │
                            ▼
                         result


          ┌────────────────────────────────────┐
          │          moe-pagecache             │
          │                                    │
          │ page table / LRU / residency       │
          │ CPU / GPU / pinned / NVMe          │
          └────────────────────────────────────┘
```

This is the architectural direction for the `moe-hpcache` branch.

## HP-cache change boundary

HP-cache must keep changes to the llama.cpp core small. Core changes are limited to wiring, argument parsing, plan handoff, and the narrow interfaces needed to invoke HP-cache. HP-cache behavior belongs in module-specific code:

- `moe-hpcache/hpcache-io.cpp` and `moe-hpcache/hpcache-io.h` own direct I/O, page alignment, bounded read buffers, and asynchronous requests.
- `moe-hpcache/hpcache-placement.cpp` and `moe-hpcache/hpcache-placement.h` own CPU and GPU expert residency decisions.
- `moe-hpcache/hpcache-scheduler.cpp` and `moe-hpcache/hpcache-scheduler.h` own CPU/GPU decode dispatch, routed expert work, partial outputs, and merge ordering.
- `moe-hpcache/hpcache.cpp` and `moe-hpcache/hpcache.h` own plan parsing, validation, and runtime state.

Do not add HP-cache policy to shared `llama-mmap.*`, generic model-loading code, or backend implementations when a module-level interface can carry the operation. Any required core change must remain a small adapter around an HP-cache module.

The immediate next step is to **walk the current llama.cpp source corresponding to #14514 and #24528 and identify the exact insertion points for HP**, rather than designing an abstraction independently of the existing implementation.

---

# 17. HP-cache placement and decode plan

This section records the intended first HP-cache policy. It is a design target; the current fork does not yet implement the placement, PLE reader, or CPU/GPU decode scheduler.

## 17.1 Model and expert placement

Keep a complete copy of the dense model weights in each participating CUDA or Vulkan device's memory and another in CPU RAM. CPU RAM holds the authoritative full-context KV cache. Allocate a bounded KV page pool and residency map on each participating GPU before placing experts. On a GPU KV miss, fill the required page from the CPU copy. KV writes update the CPU copy and any resident GPU page so that later hits remain valid.

Strata's INT8 QSA page covers four cells of one layer and is 4,224 bytes for K and V together. That size depends on Strata's tensor geometry and INT8 representation. Derive the HP-cache page bytes from the selected GGML K/V types and model geometry; do not use 4,224 bytes for `q5_0` without validating its layout. The GPU page pool must hold every distinct page needed by one attention operation, or the operation must use a defined fallback.

Load the placement plan with `--moe-hpcache=<plan.json>`. The plan records the model identity, context size, K and V cache types, participating devices, each device's KV page-pool capacity, per-device memory estimates and expert budgets, resident `(layer, expert)` entries, host placement, and the PLE GGUF location and I/O policy. The plan is the source of these settings unless an explicit expert-cache budget override is supplied. Validate that the plan matches the loaded model and available devices before allocating memory.

Set per-device expert-weight budget overrides with `--moe-hpcache-cuda-expert-cache-mib=<int1,int2,...>` for CUDA0, CUDA1, and so on, and `--moe-hpcache-vulkan-expert-cache-mib=<int1,int2,...>` for Vulkan0, Vulkan1, and so on. The first integer sets the budget for device 0 of that backend, the second for device 1, and so on. Each integer is a MiB limit. A supplied list replaces that backend's expert budgets from the plan; it does not change the other backend's budgets or the host placement algorithm. Recompute `(layer, expert)` residency after applying an override. Place as many ranked experts as fit across those device budgets and keep the nonresident experts in CPU RAM. The host placement follows the fixed algorithm and has no expert-cache budget parameter. If no expert fits on any GPU, report that no CPU/GPU placement is available. Each device budget applies to expert weights after its dense weights and bounded KV page pool have been allocated; allocation must also leave room for compute buffers and other runtime state. Count the full KV copy against CPU RAM and the resident KV pool against each GPU's memory.

For the Qwen3.8 128K `q5_0` planning case on an 8 GiB CUDA0, the proposed 80% pool leaves 3.2 GiB after 3.2 GiB of dense weights. Subtract the measured GPU KV page pool, GPU indexer and other context state, and the required compute reserve before sizing expert residency. The earlier 1.86 GiB estimate assumes the full 128K context allocation is on the GPU; it is not the paged-KV estimate. See `docs/strata-hpcache-calibration.md` for the calculation and its assumptions.

The calibration tool accepts `-ctx` with a token count or `k`/`kib`/`m`/`mib` suffix, plus `-ctk` and `-ctv`. It owns exactly 80% of each device's total memory for cacheable state, keeps the full measured KV allocation on CPU, reserves 50% of the measured KV allocation on each GPU, and reports the remainder after dense weights as expert capacity. Free-memory and compute estimates are diagnostics only. This output is a capacity estimate; runtime placement must still check live allocations.

CUDA and Vulkan entries that report the same PCI device ID are backend views of one physical device. Capacity accounting and active-work assignment select one owner backend for that device; they do not add the two backend memory pools.

## 17.2 PLE storage

Read PLE data from its GGUF file with direct I/O and asynchronous requests. Use 8 KiB logical pages for PLE reads and an explicit bounded buffer for pages in use. Do not memory-map PLE. Resolve each requested PLE row through the actual GGUF tensor offset and keep its buffer valid until all consumers finish. Align physical reads and buffers to the direct-I/O requirements of the platform; adjacent logical pages may share a larger physical read.

## 17.3 CPU and GPU decode

Add a scheduler that uses the CPU and participating CUDA or Vulkan backends during token decode. Resolve GPU KV page misses from the authoritative CPU cache before dependent attention reads. For each MoE layer, obtain the routed expert IDs at execution time. GPU-resident experts run on their assigned device while CPU workers run nonresident experts. Combine their weighted outputs before the next dependent operation. The scheduler must preserve the model's output semantics. This plan does not use MTP.

CPU and GPU expert work can overlap within one MoE layer when the routed set includes both resident and nonresident experts. Decode still follows the dependency order between layers. Measure transfers, synchronization, and output combination against a single-backend decode.

## 17.4 Parameter names

| Parameter | Purpose | Status |
| --- | --- | --- |
| `-m`, `--model` | Select the GGUF model named by the plan. | Existing llama.cpp option |
| `--moe-hpcache=<plan.json>` | Enable HP-cache, load its placement plan, and supply the model path when `--model` is omitted. | Plan schema validation implemented; placement pending |
| `--moe-hpcache-cuda-expert-cache-mib=<int1,int2,...>` | Override the plan's expert-weight budgets in MiB for CUDA0, CUDA1, and later CUDA devices. | CLI parsing implemented; placement pending |
| `--moe-hpcache-vulkan-expert-cache-mib=<int1,int2,...>` | Override the plan's expert-weight budgets in MiB for Vulkan0, Vulkan1, and later Vulkan devices. | CLI parsing implemented; placement pending |

The plan also holds the context size, K/V cache types, and GPU resident KV page capacity. Existing `--ctx-size`, `--cache-type-k`, and `--cache-type-v` options must agree with those plan values when supplied. The host placement algorithm, CPU handling of nonresident experts, and 8 KiB direct, asynchronous PLE reads are design rules here. No host expert-cache budget parameter is needed.

The current CLI slice checks that the plan path is readable, validates the generated capacity-plan schema, parses non-negative comma-separated budget lists, and stores plan metadata in `common_params`. Runtime initialization validates the plan architecture, context, and K/V types against the loaded model. Expert and KV tier allocation and decode scheduling remain separate work.

`moe-hpcache/hpcache.cpp` now resolves the validated plan against live ggml backend names during common model initialization. It selects the plan's `capacity_owner` entry for each physical backend, applies CUDA/Vulkan MiB overrides, requires an authoritative CPU entry, and publishes the resulting runtime placement state. Expert tensor allocation, CPU expert storage, KV page allocation, and decode scheduling consume this state in the next implementation stage.

HP-cache must not use the loaded model's mmap. Runtime initialization selects direct I/O. The CPU cache owns the complete expert set when the calibrated CPU budget is large enough, as it is for the 128K plan. Direct I/O fills that explicit CPU cache through bounded 8 KiB page batches. The placement stage copies the selected expert ranges into GPU resident slots. A sparse CPU source is only needed when a future plan cannot fit all experts in its CPU budget.

The current runtime wiring enforces the CPU expert placement and selects direct-I/O loading for HP-cache. The HP-cache direct-I/O reader is isolated in `moe-hpcache/hpcache-io.*`; shared `llama-mmap.*` loader code remains unchanged. GPU expert slots are not connected to `MUL_MAT_ID` execution yet; that requires splitting the layer expert tensors by routed expert and adding the corresponding merge scheduler.

Qwen4Exp now has a narrow optional MoE dispatch factory hook in the graph parameters. A null factory keeps the existing `build_moe_ffn` path. HP-cache can register a factory through the model/context wiring, while its routing and refill policy remains in the module-specific scheduler and I/O code.

### Expert paging policy

When the CPU pool cannot hold the complete expert set, it is a bounded LRU cache. A routed expert group is touched when it is used by a decode step. A miss schedules a direct-I/O read for the group's 8 KiB pages. If the cache is full, the scheduler evicts the least recently used unpinned group before installing the new group. Groups referenced by an in-flight CPU or GPU operation are pinned until their partial output is merged. The GGUF files remain the authoritative backing store; CPU and GPU resident copies are disposable cache entries.

GPU and CPU misses have different handling. If the router selects an expert that is not resident on the GPU, that expert is assigned to the CPU for this token. The GPU miss does not block on a GPU refill. If the expert is also absent from the CPU cache, the CPU scheduler reads it from GGUF through `hpcache-io`, installs it using the LRU policy, and executes the expert on the CPU. The CPU partial output is then merged with GPU partial outputs using the router weight for that expert.

---

# 18. Strata baseline: placement and hybrid scheduler

This section records the behavior implemented in the Strata reference engine at `C:/Users/suresh/devspace/github/Strata`. It is the baseline for HP-cache. The llama.cpp implementation may change the file format, page size, backends, or graph integration, but it must preserve the ownership and synchronization rules unless a variant explicitly says otherwise.

## 18.1 Strata expert placement

Strata keeps the complete expert set in host memory. Its `FileExpertSource` reads a canonical expert pack and exposes each `(layer, expert)` blob by a checked offset. A resident host arena can be pinned and mapped for direct GPU reads; when pinning the full complement is not possible, Strata falls back to ordinary resident memory or the mapped file. The host copy remains available for CPU execution.

The GPU tier is an `ExpertCache` containing fixed slots and a checked `(layer, expert) -> slot` table. Startup checks the requested arena against live free VRAM, allocates one device arena, zeroes it, and fills selected slots from the host source. The cache records slot offsets for variable-size native expert blobs. A fill is asynchronous on the decode stream; startup also has blocking and queued fill paths followed by verification.

The initial resident set comes from a profile file ranked by routing frequency. The profile contains model dimensions, a slot count, and ranked `(layer, expert)` pairs. Strata validates the model dimensions and pair bounds before accepting it. Its per-layer admission variant divides the GPU slots among layers so arrival order cannot consume the whole cache with the first layers encountered. The baseline cache does not evict during normal dispatch. A missing expert is admitted only if a free slot remains; otherwise it stays a CPU miss.

Strata also has an adaptive tier variant. It accumulates decayed routing usage by `(layer, expert)` and, at configured round boundaries, replaces less useful GPU entries with more frequently routed missing entries. This is a policy variant, separate from the fixed profile placement. The host arena and CPU fallback remain unchanged during a swap.

For multiple GPUs, Strata supports two different arrangements:

- In layer split mode, contiguous layer ranges are assigned to devices. Each device keeps dense/state data and an expert cache for its own layers. A token crosses devices once per verify window through pinned host handoff buffers. The host expert arena is shared, and the CPU computes each routed pair absent from the owning device.
- In remote expert mode, CUDA0 keeps the main model state and up to three secondary devices own additional expert entries. Ranked pairs not resident on CUDA0 are assigned to secondary caches, either in a stripe across devices or by layer. Each secondary GPU returns packed expert rows through private pinned buffers before the layer combine.

## 18.2 Strata per-layer dispatch

At a routed MoE layer, Strata first deduplicates the routed `(token, expert)` entries into distinct experts. It classifies every entry as:

1. a VRAM hit, with a pointer to the resident slot;
2. an optional PCIe GPU miss, selected from host-pinned experts when the configured PCIe share allows it; or
3. a CPU miss, evaluated from the host expert source.

The GPU plan is published before the CPU work begins. The plan contains grouped GPU pointers, destination rows, token indices, router weights, and counts. A copy engine may stage pinned host experts into reusable device buffers, or a copy kernel may read the mapped host alias. Secondary GPU plans use their own pinned input, metadata, and output buffers.

The CPU side builds one job per distinct missing expert and reuses the activation for all routed tokens that selected that expert. The CPU receives the CPU subset of the router's `(expert_id, router_weight)` list; it does not receive the GPU subset as CPU work. The router weight stays associated with its expert ID and destination row until the final combine. The `ExpertPool` uses an epoch-tagged, lock-free batch: workers claim jobs with a CAS counter, compute, increment a completion count, and re-park. The host can also claim jobs. The pool waits for both all jobs to finish and all workers to re-park before reusing the batch description. For verify windows, one expert job carries the routed tokens for that expert, so its weight rows are read once for multiple tokens. When only a few CPU experts remain, Strata can split each expert's rows across workers to keep the memory bandwidth occupied.

## 18.3 Strata hybrid decode scheduler

The single-token scheduler is a two-graph pipeline around a captured layer graph:

```text
GPU: layer pre -> router -> publish doorbell -> GPU expert hits/shared work -> wait for CPU flag -> combine -> layer post
CPU:                         observe doorbell -> compute nonresident experts -> publish completion flag
```

The doorbell is mapped pinned memory with fixed device and host addresses. A captured GPU kernel writes the sequence number and routed IDs/weights into that memory. The host polls the sequence and performs a nonblocking CUDA stream query periodically so the driver submits work on systems where a host memory spin alone does not flush the GPU queue. The host then runs the CPU misses, executes memory fences, and writes the completion flag. The captured graph waits on that flag before copying the CPU result rows into the device result buffer and combining GPU and CPU contributions.

This preserves the layer dependency: the next layer does not consume the residual until the current layer's combine is complete. CPU and GPU work overlaps inside one layer, but the scheduler does not use a one-layer stale result. Strata's graph is split into `pre` and `post` specifically so the current layer's routed IDs, weights, and expert outputs are joined before its residual write.

The verify-window path uses the same classification contract without the single-token doorbell graph. It publishes the GPU hit/PCIe plan, starts the GPU work, runs the CPU pool for remaining experts, waits for remote GPU results, and merges rows by their original routed destinations. Duplicate routed experts are computed once per window and their output is written to every destination that selected them.

Strata's KV streaming is a separate residency system. CPU pinned memory is authoritative and contains the full K/V history. Each GPU layer has a page table, a fixed page-slot pool, and a host-to-device miss path. Before attention reads, the selected logical pages are resolved: resident pages get their use stamp, missing pages claim slots and are copied from the host cache. A CLOCK-style sweep chooses victims while protecting pages used by the current selection. Appends always update the CPU copy and update the GPU slot only when that page is resident. In the QSA INT8 geometry, one page is four cells and 4,224 bytes for K and V together; HP-cache must derive its own page bytes for GGML `q5_0`.

## 18.4 Baseline invariants for HP-cache

The following are the Strata behaviors that the first HP-cache implementation should retain:

- CPU owns a complete authoritative copy of experts and streamed KV.
- GPU residency is bounded and represented by explicit tables; a cache miss has a defined CPU or host-to-device path.
- Routing is deduplicated before dispatch, then partitioned into backend-specific expert ID subsets. Each subset retains the router weight belonging to each expert ID, and output rows retain their original token/expert destinations.
- GPU work is published first so CPU misses can run concurrently; the merge waits for both sides.
- Layer dependencies remain ordered. Parallelism is within a layer and across independent devices, not by feeding a previous layer's result into the next layer.
- Placement, source I/O, backend execution, and synchronization are separate responsibilities.

Our planned variants are the JSON plan and comma-separated CUDA/Vulkan budget overrides, GGML K/V page sizing, DirectIO plus asynchronous 8 KiB PLE reads, and a llama.cpp backend implementation of the same hybrid scheduling contract. MTP remains disabled for this mode, so the scheduler must support ordinary decode with at least one CPU-resident expert whenever the GPU does not hold the full routed set.

## 18.5 Decode dependency and look-ahead boundary

The CPU miss work is the last expert contribution for the current layer, not an independent path to the next layer. The GPU can finish resident experts, shared work, buffer preparation, and prefetches while the CPU computes its subset. The next layer starts only after the current layer's GPU and CPU expert rows have been merged and the residual state has been written.

For autoregressive single-token decode, the next token cannot start until the current token has passed through the final layer, produced logits, and been sampled. The first HP-cache mode does not use MTP or another draft model, so it does not promise single-sequence look-ahead. Parallel work across independent sequences and verify-window batches is allowed. Speculative look-ahead is a later feature with its own draft, verification, KV rollback, and scheduling rules.

KV ownership does not change this dependency. Attention creates the current K/V before the MoE split. CPU RAM receives or owns the authoritative copy, and the GPU updates its resident page when present. The CPU expert backend does not create a second K/V result. A later attention operation may read the new entry only after the authoritative write and required device visibility are complete.

