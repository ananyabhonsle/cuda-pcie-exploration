![C++](https://img.shields.io/badge/C++-17-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.x_%2F_13.x-green.svg)
![Status](https://img.shields.io/badge/Status-Phase_3-yellow.svg)

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
*   **Next:** Phase 4 - Latency harness: audio-sized chunks, a stand-in GPU kernel per chunk, and per-chunk p50/p99 latency. Then sync/async `cudaMemcpy` bridges and a mutex-queue baseline to compare against the lock-free zero-copy design.

---

**Phase 3 Results** (1,000,000 items)

| | RTX 3050 Ti Laptop (Windows 11, CUDA 12.9) | Tesla T4 (Google Colab, Linux, CUDA 13.0) |
|---|---|---|
| 3a: 1-thread consumer | ~157 ms (~80 us/batch) | ~194 ms (~99 us/batch) |
| 3c: 32-thread warp consumer | ~30 ms (~15 us/batch) | ~38 ms (~19 us/batch) |
| Warp speedup | 5.3x | 5.2x |

Laptop figures are typical of 3-5 runs; T4 figures are from one run. The T4 VM is slower even though Linux has no WDDM launch batching, which points to PCIe round-trip latency, not the OS, as the main cost per batch.

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
```
