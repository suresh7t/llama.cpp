# moe-hpcache — Heterogeneous Parallel (HP) Execution

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

The immediate next step is to **walk the current llama.cpp source corresponding to #14514 and #24528 and identify the exact insertion points for HP**, rather than designing an abstraction independently of the existing implementation.
