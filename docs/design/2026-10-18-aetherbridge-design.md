# AetherBridge: Design

*Zero-copy, lock-free batching for PCIe-bound audio-LLM inference.*

Status: draft, 2026-10-18. Covers Phases 4-8. Phases 1-3 are complete and merged (see README).

Labels used below: **[decided]** = agreed with the project contributor; **[my call]** = chosen while writing this, open to change; **[builder]** = left to whoever implements it.

---

## 1. Intent

A speech-capable LLM consumes audio as a steady stream of small chunks (tens of milliseconds each). Every chunk has to move from CPU memory to the GPU over PCIe before the model can use it. For small chunks the cost of that hand-off is dominated by fixed per-transfer overhead (PCIe round trips, driver calls, thread hand-offs), not by bandwidth, and in a streaming system the **tail latency** (p99) matters more than the average: one slow chunk stalls the stream.

AetherBridge is the CPU→GPU hand-off layer, built as a **benchmark study**: several bridge designs, one shared measurement harness, and a write-up that answers:

> Does a lock-free queue plus an asynchronous or zero-copy transport beat a mutex queue plus synchronous copies on p50/p99 latency and throughput for audio-sized chunks, and by how much?

**[my call]** The deliverable is the benchmark and the write-up, not a reusable library with a stable API.

## 2. Goals, non-goals, anti-goals

**Goals**
1. Five bridge variants (Section 4) that differ only in the queue and the transport.
2. One harness that measures every variant the same way: per-chunk end-to-end latency (p50, p99, p99.9, max) and throughput.
3. An integrity check on every run: every chunk is processed exactly once and uncorrupted (order may differ only in the async variants).
4. Results on two platforms: RTX 3050 Ti Laptop (Windows 11) and Tesla T4 (Google Colab, Linux).
5. CSV output, charts, and a README write-up of the results.

**Non-goals [my call]**
- Speech-recognition accuracy or real model quality.
- Multiple producers, multiple concurrent audio streams, or multiple GPUs.
- GPUDirect, NVLink, or any transport besides PCIe via the CUDA runtime.
- Python bindings or a library API.

**Anti-goals (would make the project a failure even if it runs)**
- Comparing variants that differ in more than one thing at a time (chunk size, GPU work, thread count, or measurement method must be identical across variants).
- Reporting averages only, so tail latency is hidden.
- Claims in the README that the data does not support.
- Changes to the user's system to make numbers look better (e.g. registry edits to the Windows GPU timeout).

## 3. Constraints

From Phases 1-3 and the test machines:

| Constraint | Consequence |
|---|---|
| C-style C++17, compiled with `nvcc`; must build with CUDA 12.9 + MSVC 14.44 (Windows) and CUDA 13.0 (Colab). | No compiler-specific extensions beyond what Phase 3 already uses. |
| Only the CUDA compiler and runtime are installed locally (no cuFFT, cuBLAS, or other libraries); drive C has ~5 GB free. | The stand-in GPU kernel must be plain CUDA. No new large dependencies. |
| Windows (WDDM) kills any kernel running longer than ~2 s and resets the GPU (`kernelExecTimeout = 1`). | Long-running (persistent) kernels must be split into episodes of at most ~1 s (Section 6.4). |
| Windows queues kernel launches until something flushes them. | Every launch that must start promptly is followed by `cudaStreamQuery`. Windows numbers include this overhead; that is reported, not hidden. |
| The laptop GPU has `hostNativeAtomicSupported = 0`. | Shared CPU/GPU state uses only atomic loads and stores (system-scope `cuda::atomic_ref`), never read-modify-write. |
| A GPU cannot take a CPU `std::mutex`. | There is no mutex + zero-copy variant (Section 4). |
| Phase 3 measured ~1 us per GPU read of host memory and ~15-19 us per batch (warp consumer). | Expected unloaded zero-copy latency is a few microseconds; anything far off that is a harness bug until shown otherwise. |

## 4. The variants

Two dimensions: the **queue** between the CPU producer and the bridge, and the **transport** that gets each chunk onto the GPU.

|  | sync copy | async copy | zero-copy |
|---|---|---|---|
| **mutex queue** | V1 | V2 | (impossible) |
| **lock-free queue** | V3 | V4 | V5 |

(The mutex row is run with both waiting styles, Section 4.1, so in practice there are seven configurations per workload.)

**[decided]** Five cells, not six: in zero-copy the GPU itself is the consumer of the queue, and a GPU cannot lock a CPU mutex (or build one over PCIe without host-native atomics). The write-up should say this explicitly; it is a finding in itself.

### 4.1 Queues

Both queues are single-producer, single-consumer, hold the same chunk type, and have the same capacity **[my call: 64 chunks; builder may tune but must use the same value for both]**.

- **Lock-free:** the Phase 2/3 ring buffer, generalised from `int` slots to chunk slots. Same acquire/release discipline.
- **Mutex:** a fixed-capacity ring protected by `std::mutex`. No lock-free tricks inside it.

For V1-V4 the queue lives in ordinary (pageable) CPU memory, like an application's own audio buffers; for V5 it lives in mapped pinned memory, because the GPU reads it directly.

**Waiting style [my call].** How a thread waits when the queue is empty or full changes latency as much as the locking does, so it is its own setting, recorded in the CSV:
- `spin`: re-check in a loop (with a pause instruction). The lock-free queue always spins, as in Phases 2-3.
- `block`: sleep on a `std::condition_variable` until notified. Mutex queue only.

The mutex queue supports both. The **headline comparison is lock-free/spin vs. mutex/block**, the conventional baseline. **Mutex/spin** is also run, so the write-up can separate the cost of locking from the cost of sleeping and waking.

### 4.2 Transports

- **Sync copy (V1, V3):** a CPU bridge thread pops a chunk, copies it into a pinned staging buffer, calls `cudaMemcpy` to the GPU, launches the processing kernel, and waits for it to finish before popping the next chunk.
- **Async copy (V2, V4):** the bridge thread pops a chunk, copies it into one of K pinned staging slots, issues `cudaMemcpyAsync` and the kernel launch on that slot's own stream, and moves on without waiting. **[my call: K = 3 slots, one CUDA stream per slot, used in rotation.]** One stream per slot is required, not a detail: operations on a single stream run strictly in order, so with one stream chunk n+1's copy would wait for chunk n's kernel and nothing would overlap. A CUDA event per slot tells the bridge thread when that slot can be reused.

In both copy transports the bridge copies the chunk out of the queue into pinned staging before the GPU copy. **[my call]** This keeps the queue independent of the transport (a queue slot is free as soon as it is popped) and lets `cudaMemcpy` run from pinned memory, as a real application would.
- **Zero-copy (V5):** no bridge thread. The queue lives in mapped pinned memory and a persistent GPU kernel consumes it directly, as in Phase 3 (warp consumer).

### 4.3 Threads per variant

| Variant | CPU threads | GPU |
|---|---|---|
| V1-V4 | producer, bridge, ack observer | one short kernel launch per chunk |
| V5 | producer, ack observer | one persistent kernel per episode |

V5 has one fewer CPU hop. That is inherent to zero-copy, not a bias in the benchmark, and the write-up should say so.

## 5. Workload

### 5.1 Chunks

**[my call]** Audio format: 16 kHz, mono, `float32`. Chunk duration sets the size:

| Chunk | Samples | Bytes |
|---|---|---|
| 10 ms | 160 | 640 |
| **20 ms (default)** | 320 | 1,280 |
| 40 ms | 640 | 2,560 |
| 80 ms | 1,280 | 5,120 |

Each chunk also carries a 64-bit sequence number. Sample values are synthetic and deterministic (generated from the sequence number) so the GPU can verify them without the CPU sending a separate checksum. **[builder]** Exact generator.

### 5.2 Stand-in GPU kernel

**[decided]** Phases 4-7 use a stand-in kernel, not a real model, so the benchmark measures the bridge, not model performance.

The kernel processes one chunk with **one warp (32 threads) in every variant** **[my call]**, so GPU-side parallelism is identical everywhere. Per chunk it:
1. verifies the chunk's samples against the expected pattern for its sequence number, and counts mismatches;
2. does a configurable amount of arithmetic per sample to stand in for feature extraction / model work;
3. writes the chunk's sequence number to that chunk's ack slot (Section 6.1).

The same `__device__` function does this in all variants: V1-V4 launch it per chunk; V5 calls it from the persistent kernel.

**[my call]** Three work levels, calibrated on the laptop GPU by the builder:

| Level | Target GPU time per chunk |
|---|---|
| `none` | ~0 (verification only) |
| `light` | ~10 us |
| `heavy` | ~200 us |

Async copy only pays off when there is GPU work to overlap with, so the work level is a real experimental variable, not a detail.

### 5.3 Producer pacing

- **Unpaced:** the producer pushes chunks as fast as the queue accepts them. Measures **throughput**. Latency under unpaced load is mostly queueing time and is reported but labelled as such.
- **Paced:** the producer pushes one chunk every P microseconds, with P much longer than any variant's service time, so queues stay nearly empty. Measures **unloaded hand-off latency**, the headline latency number. **[my call: default P = 1,000 us]** Real-time audio pacing (one 20 ms chunk every 20 ms) is a trivially light load and is not needed separately.

**[builder]** Pacing by spin-wait on `steady_clock`, not `sleep_for`, since sleeps on Windows have ~1 ms granularity.

## 6. Measurement

### 6.1 Latency on one clock

CPU and GPU clocks are not synchronised, so latency is measured entirely on the CPU's `steady_clock`:

1. The producer records `t_send[seq]` immediately before pushing chunk `seq`.
2. After the GPU has processed chunk `seq`, it writes `seq` into **ack slot `seq % A`**, an array of A slots in mapped pinned memory (system-scope release store). **[my call: A = 4,096]**, far more than the number of chunks that can be in flight (queue capacity + K).
3. An **ack observer** CPU thread spins over the slots of all chunks still in flight; when a slot shows the expected `seq`, it records `t_ack[seq]`.
4. Latency for the chunk = `t_ack[seq] - t_send[seq]`.

Per-chunk slots, not a single "last finished" word, because with K streams the async variants can finish chunks **out of order**; a single word would lose acknowledgments. V1, V3 and V5 finish in order, but use the same mechanism so the measurement is identical everywhere.

This includes the ack's trip back over PCIe. That cost is the same in every variant, so comparisons stay fair; the write-up should note that absolute numbers include it.

### 6.2 What each run reports

| Field | Meaning |
|---|---|
| `variant`, `queue`, `wait`, `transport` | which cell, and `spin` or `block` |
| `queue_capacity`, `staging_slots` | so tuned values are always on record |
| `chunk_ms`, `chunk_bytes`, `work`, `pacing_us` | workload (`pacing_us = 0` means unpaced) |
| `platform`, `gpu`, `cuda` | where it ran |
| `chunks`, `warmup` | samples measured / discarded |
| `p50_us`, `p99_us`, `p999_us`, `max_us`, `mean_us` | latency |
| `chunks_per_s`, `mb_per_s` | throughput |
| `mismatches`, `stalled` | integrity and watchdog results |

One CSV row per run, appended to a results file. **[builder]** Optional raw per-chunk latency dump for plotting distributions.

### 6.3 Sample sizes

**[my call]** Discard the first 500 chunks as warm-up. Measure at least 20,000 chunks per configuration so p99 rests on ~200 samples and p99.9 on ~20; repeat each configuration 3 times and report the median run plus the spread.

Expected cost: a paced run of 20,000 chunks at 1,000 us takes ~20 s, so the full paced matrix (7 configurations x 4 chunk sizes x 3 work levels x 3 repeats = 252 runs) is about 1.5 hours per platform; unpaced runs are much shorter. **[builder]** The runner should support running a subset, and resuming.

### 6.4 Windows kernel timeout (episodes)

V5's persistent kernel would exceed the ~2 s Windows limit on any run of useful length. **[my call]** On all platforms, V5 runs in **episodes**: the persistent kernel is launched, consumes chunks for at most ~1 s (or a fixed chunk count), and exits; the next episode is launched. Chunks sent while no kernel is running would distort latency, so the producer must not send across an episode boundary: it pauses, the old kernel exits, the new kernel starts, then sending resumes. Episode gaps are excluded from measurements: no latency samples are taken across a boundary, and throughput is computed over active time only, with the number of episodes recorded. Using episodes on Linux too keeps the method identical across platforms.

V1-V4 launch one short kernel per chunk and are not affected.

### 6.5 Noise control

**[builder]** Laptop runs on mains power with other applications closed; the CSV records the platform so runs are never mixed. Pinning threads to cores is optional; if used, record it.

### 6.6 Safety

Every variant keeps the Phase 3 stall watchdog: any thread or kernel that sees no progress for 1 s gives up and the run reports `stalled = 1` instead of hanging or triggering a GPU reset. A stalled or mismatched run is reported and excluded from charts.

## 7. Code layout

**[my call]**, matching the existing one-file-per-phase style while sharing code across variants:

```
src/aetherbridge/
  common.h        chunk type, CUDA_CHECK, timing, stats (percentiles), CSV writer
  queues.h        lock-free SPSC queue, mutex queue (same interface)
  kernels.cuh     stand-in __device__ processing function, ack write
  bench.cu        main: parses flags, runs one configuration, appends a CSV row
scripts/
  run_matrix.py   runs every configuration, collects CSV, makes charts
results/          CSVs and charts committed with the write-up
```

`src/phase1..3` files stay unchanged as the historical record.

**[my call]** One binary selected by flags, e.g.
`bench --queue lockfree|mutex --transport sync|async|zerocopy --chunk-ms 20 --work none|light|heavy --pace-us 1000 --chunks 20000 --csv results/run.csv`.
**[builder]** Exact flag names, and rejecting `--queue mutex --transport zerocopy` with a clear message.

**[my call]** The matrix runner and charts use Python 3 with matplotlib (Python 3.12 is installed locally; Colab has both). It only launches the binary and plots; no measurement logic lives in Python.

## 8. Phases

Each phase is its own branch and PR into `main`, tested on the laptop and, from Phase 5 on, on Colab before merging.

| Phase | Builds | Done when |
|---|---|---|
| **4** | `common.h`, `kernels.cuh`, the ack/latency harness, chunk workload, stats and CSV; V5 (zero-copy) ported from Phase 3 to chunks, with episodes | V5 reports p50/p99/p99.9 and throughput for 20 ms chunks at all three work levels, zero mismatches, on the laptop and the T4 |
| **5** | Lock-free queue in `queues.h`; sync and async copy transports (V3, V4) | V3 and V4 pass integrity and report the same CSV fields as V5 |
| **6** | Mutex queue; V1 and V2 | All five variants run from the same binary |
| **7** | `run_matrix.py`, the full matrix (7 configurations, i.e. the 5 variants with the mutex row in both waiting styles, x 4 chunk sizes x 3 work levels x paced/unpaced) on both platforms, charts, README write-up | Results committed; the README answers the question in Section 1, with the data |
| **8** *(optional)* | Feed chunks into a real audio model on the GPU | Separate decision before starting (Section 10) |

## 9. Verification

- **Integrity:** zero mismatches and every sequence number acknowledged exactly once, on every run. V1, V3 and V5 must also acknowledge in order.
- **Sanity checks the harness enforces or reports:** `p50 <= p99 <= p99.9 <= max`; number of latency samples equals `chunks`; V5 unloaded latency with `work = none` is in the low microseconds, consistent with Phase 3's ~1 us per PCIe read.
- **Negative tests (kept from Phase 3):** no producer, no consumer, and no kernel each produce a clean `stalled` result within ~1.5 s.
- **Cross-platform:** every phase from 5 on is run on both the laptop and Colab before its PR is merged.

## 10. Open items

- **Phase 8 scope:** whether to run a real model, which one (e.g. a Whisper encoder), and with what runtime. This needs new dependencies and disk space, so it is decided separately after Phase 7.
- **Naming:** the repository and README are still titled "CUDA PCIe Exploration". Renaming to AetherBridge is the repository owner's decision.

## 11. Left to the builder

- Exact flag names, file splits within `src/aetherbridge/`, and internal function names.
- The synthetic sample generator and the arithmetic used for work levels, provided the work-level targets in Section 5.2 are met and documented.
- How percentiles are computed (sorting is fine at these sample sizes).
- Chart styling and which charts beyond: latency CDF per variant, p99 vs chunk size, and throughput per variant.
- Tuning queue capacity and K, provided all variants use the same values and the values are recorded in the CSV.
