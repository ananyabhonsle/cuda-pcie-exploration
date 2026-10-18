![C++](https://img.shields.io/badge/C++-17-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.x_%2F_13.x-green.svg)
![Status](https://img.shields.io/badge/Status-Phase_4-yellow.svg)

**CUDA PCIe Exploration**
*Learning low-level CUDA memory management by characterizing host-to-device transfer latency.*

---

**Current Status**
*   **Phase 1 (Complete):** Validated `cudaHostAllocWriteCombined` allocation and benchmarked CPU write/read throughput on write-combined vs. pinned vs. pageable memory.
*   **Phase 2 (Complete):** Validated lock-free SPSC ring buffer with `alignas(64)` cache-line padding. Successfully transferred 1,000,000 integers with zero data corruption.
*   **Phase 3 (Complete):** Moved the ring buffer into mapped pinned memory with a GPU kernel as the consumer (zero-copy: no `cudaMemcpy`). The CPU producer and GPU consumer sync over PCIe with system-scope `cuda::atomic_ref` loads/stores; the GPU drains the buffer in batches. 1,000,000 integers transferred with zero data corruption on both Windows and Linux.
    *   **3a - 1-thread consumer:** baseline.
    *   **3b - batch-size log:** batches lock into alternating pairs that sum to a full buffer (e.g. `337 686`, `294 729`): each batch is exactly what the producer refilled while the GPU read the previous one. Seen on both GPUs below.
    *   **3c - 32-thread warp consumer (`--warp`):** ~5x faster, because a warp fetches 128 bytes per PCIe request instead of 4.
*   **Phase 4 (laptop done, T4 pending):** Latency harness plus the zero-copy variant (V5: lock-free queue, persistent GPU kernel reading the ring over PCIe). Audio-sized chunks (20 ms = 1280 B), a stand-in kernel at three work levels (none, light, heavy), per-chunk p50/p99 latency, the GPU work time measured inside the kernel, and a CSV row per run. Every run verifies each chunk's contents and header (0 mismatches), checks that every published chunk was consumed exactly once, and detects stalls. The Colab T4 run is still to do.
*   **Next:** Phase 5 - the sync and async `cudaMemcpy` bridges (`--transport sync|async`). Phase 6 adds the mutex-queue baseline to compare against the lock-free zero-copy design.

---

**Phase 3 Results** (1,000,000 items)

| | RTX 3050 Ti Laptop (Windows 11, CUDA 12.9) | Tesla T4 (Google Colab, Linux, CUDA 13.0) |
|---|---|---|
| 3a: 1-thread consumer | ~157 ms (~80 us/batch) | ~194 ms (~99 us/batch) |
| 3c: 32-thread warp consumer | ~30 ms (~15 us/batch) | ~38 ms (~19 us/batch) |
| Warp speedup | 5.3x | 5.2x |

Laptop figures are typical of 3-5 runs; T4 figures are from one run. The T4 VM is slower even though Linux has no WDDM launch batching, which points to PCIe round-trip latency, not the OS, as the main cost per batch.

---

**Phase 4 Results** (V5, 20 ms chunks, 20000 chunks + 500 warmup, RTX 3050 Ti Laptop, Windows 11, CUDA 12.9)

| Work | Pacing | p50 | p99 | p99.9 | max | gpu_work |
|---|---|---|---|---|---|---|
| none | 1000 us | 7.3 us | 400.4 us | 3216.2 us | 17895.3 us | 6.5 us |
| light | 1000 us | 15.1 us | 445.9 us | 2372.8 us | 11457.8 us | 14.9 us |
| heavy | 1000 us | 166.4 us | 570.3 us | 2371.8 us | 10063.9 us | 169.5 us |
| none | unpaced | 426.3 us | 900.5 us | 1237.5 us | 1240.8 us | 4.5 us |

Unpaced throughput: 143,425 chunks/s (183.6 MB/s). In the unpaced row the producer keeps the queue full, so its latency is mostly time spent waiting in the queue, not hand-off cost.

A latency sample starts when the producer begins copying a chunk into a queue slot that has space, and ends when the ack observer notices the GPU's ack. Generating the chunk's data and waiting for queue space are not included.

`gpu_work` is the mean time per chunk the kernel spent in its processing step (reading the chunk's samples over PCIe, checking them, and the stand-in work), measured inside the kernel with `globaltimer` and recorded in the CSV as `gpu_work_us`. It is the authoritative work figure for each run: the iteration counts behind light and heavy are fixed, but on this laptop the time they take varies by ~30% between sessions with GPU clock and power state. Being a mean, it also picks up rare long pauses inside the kernel. That is why heavy's mean (169.5 us) is above its p50 latency (166.4 us).

Paced `work none` p50 is ~7 us. As an estimate, not a measurement: Phase 3 measured ~1 us per GPU read of host memory, and each paced chunk needs several dependent PCIe accesses (the kernel's poll of head, the sample reads, which each lane issues in a few dependent batches, the ack write, and the observer noticing the ack), which is consistent with ~7 us. The in-kernel timing behind `gpu_work` adds about 1.3 us to this p50 (6.0 vs 7.3 us in interleaved runs of builds with and without it); at light and heavy no difference was measurable.

The tail is much longer than the median: in these paced runs p99 is 400-570 us, p99.9 is 2.4-3.2 ms and max is 10-18 ms. Tail latencies vary a lot from run to run on this laptop (a review saw p99 range from 68 to 1380 us across four runs), so single-run tails are indicative only; Phase 7 repeats each configuration 3 times. These runs were also taken with other GPU applications open, which probably widened the tails. The likely cause of the tails is Windows/WDDM scheduling or thread descheduling, but this was not confirmed. These are single runs from one laptop; the T4 run is still to do.

---

**Tech Stack**
*   **Language:** C-style C++17
*   **Compiler:** NVCC (tested with CUDA 12.9 + MSVC 14.44 on Windows, CUDA 13.0 on Linux)
*   **Target Hardware:** Discrete NVIDIA GPUs (Cloud T4 / AWS A100)
*   **Tested On:** RTX 3050 Ti Laptop GPU (Windows 11), Tesla T4 (Google Colab)

---

**Build & Run**
```bash
# Phase 1: WC memory test (needs an NVIDIA GPU)
nvcc -O2 -std=c++17 src/wc_memory_test.cu -o phase1
./phase1

# Phase 2: lock-free SPSC ring buffer (CPU only)
g++ -O2 -std=c++17 -pthread src/phase2_ringbuffer.cpp -o phase2
./phase2

# Phase 3: CPU producer -> GPU consumer over PCIe (needs an NVIDIA GPU)
nvcc -O2 -std=c++17 -arch=native src/phase3_gpu_consumer.cu -o phase3
./phase3          # 1-thread consumer
./phase3 --warp   # 32-thread warp consumer

# Phase 4: latency harness, V5 (lock-free queue + zero-copy)
# On Linux with glibc older than 2.34, add -lpthread to the nvcc commands below.
mkdir -p build
nvcc -O2 -std=c++17 -arch=native -I src src/aetherbridge/bench.cu -o build/bench
./build/bench --work none                       # paced every 1000 us, 20 ms chunks
./build/bench --work heavy --csv results.csv    # append a CSV row
./build/bench --work none --pace-us 0           # unpaced throughput
# flags: --queue --wait --transport --chunk-ms --work --pace-us --chunks --warmup --csv

# Phase 4 tests (same command per file)
for t in test_common test_config test_kernels test_harness test_zerocopy; do
  nvcc -O2 -std=c++17 -arch=native -I src tests/$t.cu -o build/$t && ./build/$t
done
```
