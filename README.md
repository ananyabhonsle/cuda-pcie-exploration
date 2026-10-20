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
*   **Phase 4 (Complete):** Latency harness plus the zero-copy variant (V5: lock-free queue, persistent GPU kernel reading the ring over PCIe). Audio-sized chunks (20 ms = 1280 B), a stand-in kernel at three work levels (none, light, heavy), per-chunk p50/p99 latency, the GPU work time measured inside the kernel, and a CSV row per run. Every run verifies each chunk's contents and header (0 mismatches), checks that every published chunk was consumed exactly once, and detects stalls. Tested on the RTX 3050 Ti laptop (Windows) and a Colab Tesla T4 (Linux).
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

**Phase 4 Results** (V5, 20 ms chunks, 20000 chunks + 500 warmup, one run per row)

RTX 3050 Ti Laptop, Windows 11, CUDA 12.9, GPU otherwise idle:

| Work | Pacing | p50 | p99 | p99.9 | max | gpu_work |
|---|---|---|---|---|---|---|
| none | 1000 us | 7.0 us | 8.5 us | 52.9 us | 373.3 us | 4.4 us |
| light | 1000 us | 15.0 us | 16.5 us | 497.9 us | 7554.0 us | 12.2 us |
| heavy | 1000 us | 159.4 us | 185.9 us | 389.8 us | 4998.5 us | 156.4 us |
| none | unpaced | 423.5 us | 454.3 us | 848.0 us | 976.1 us | 4.4 us |

Tesla T4 (Google Colab), Linux, CUDA 13.0:

| Work | Pacing | p50 | p99 | p99.9 | max | gpu_work |
|---|---|---|---|---|---|---|
| none | 1000 us | 8.3 us | 1516.7 us | 5348.7 us | 9351.6 us | 4.6 us |
| light | 1000 us | 19.6 us | 1003.0 us | 2770.6 us | 4296.9 us | 16.0 us |
| heavy | 1000 us | 193.9 us | 1478.4 us | 3537.3 us | 5437.4 us | 190.8 us |
| none | unpaced | 497.8 us | 631.5 us | 1037.5 us | 1300.7 us | 4.4 us |

Unpaced throughput: 150,588 chunks/s (192.8 MB/s) on the laptop, 127,653 chunks/s (163.4 MB/s) on the T4. In the unpaced rows the producer keeps the queue full, so latency there is mostly time spent waiting in the queue, not hand-off cost.

A latency sample starts when the producer begins copying a chunk into a queue slot that has space, and ends when the ack observer notices the GPU's ack. Generating the chunk's data and waiting for queue space are not included.

`gpu_work` is the mean time per chunk the kernel spent in its processing step (reading the chunk's samples over PCIe, checking them, and the stand-in work), measured inside the kernel with `globaltimer` and recorded in the CSV as `gpu_work_us`. It is the authoritative work figure for each run: the iteration counts behind light and heavy are the same on every GPU, so the time they take differs by GPU (heavy: ~156 us on the laptop, ~191 us on the T4) and, on the laptop, by up to ~30% between sessions with clock and power state.

Paced `work none` p50 is ~7-8 us. As an estimate, not a measurement: Phase 3 measured ~1 us per GPU read of host memory, and each paced chunk needs several dependent PCIe accesses (the kernel's poll of head, the sample reads, the ack write, and the observer noticing the ack). The in-kernel timing behind `gpu_work` adds about 1.3 us to the `work none` p50 (A/B ranges 5.9-6.3 us without it vs 7.2-7.6 us with it); at light and heavy no difference was measurable.

Tails: on the idle laptop p99 is within ~2 us of the median at every work level; an earlier laptop run with other GPU applications open had p99 of 400-570 us, so sharing the GPU matters a lot. On the T4, p99 is 1.0-1.5 ms. A likely cause, not confirmed: the Colab VM has few CPU cores, and V5 keeps two threads (producer and ack observer) spinning, so they get descheduled. The rare multi-millisecond maxima on both machines are likely OS scheduling. These are single runs; Phase 7 repeats each configuration 3 times.

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
