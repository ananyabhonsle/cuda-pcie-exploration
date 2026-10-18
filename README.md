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
*   **Phase 4 (Complete, laptop):** Latency harness plus the zero-copy variant (V5: lock-free queue, persistent GPU kernel reading the ring over PCIe). Audio-sized chunks (20 ms = 1280 B), a calibrated stand-in kernel at three work levels (none, light, heavy), per-chunk p50/p99 latency, and a CSV row per run. Every run verifies each chunk's contents (0 mismatches) and detects stalls. The Colab T4 numbers are still to be added.
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

| Work | Pacing | p50 | p99 | p99.9 |
|---|---|---|---|---|
| none | 1000 us | 9.8 us | 156 us | 827 us |
| light | 1000 us | 16.2 us | 147 us | 721 us |
| heavy | 1000 us | 158.7 us | 328 us | 1098 us |
| none | unpaced | 554 us | 590 us | 1355 us |

Unpaced throughput: ~114,000 chunks/s (146 MB/s). In the unpaced row the producer keeps the queue full, so its latency is mostly time spent waiting in the queue, not hand-off cost.

Paced `work none` p50 of ~10 us is about 5-7 dependent PCIe round trips per chunk. About 5% of chunks show 0.1-1.2 ms spikes, which drive p99; the likely cause is WDDM scheduling or thread descheduling, but this was not confirmed. These are single runs from one laptop; the T4 run is still to do.

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
