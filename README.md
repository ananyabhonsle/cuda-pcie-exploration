![C++](https://img.shields.io/badge/C++-17-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)
![Status](https://img.shields.io/badge/Status-Phase_3-yellow.svg)

**CUDA PCIe Exploration**
*Learning low-level CUDA memory management by characterizing host-to-device transfer latency.*

---

**Current Status**
*   **Phase 1 (Complete):** Validated `cudaHostAllocWriteCombined` allocation and benchmarked CPU write/read throughput on write-combined vs. pinned vs. pageable memory.
*   **Phase 2 (Complete):** Validated lock-free SPSC ring buffer with `alignas(64)` cache-line padding. Successfully transferred 1,000,000 integers with zero data corruption.
*   **Phase 3 (Complete):** Moved the ring buffer into mapped pinned memory with a GPU kernel as the consumer. The CPU producer and GPU consumer sync over PCIe with system-scope `cuda::atomic_ref` loads/stores; the GPU drains the buffer in batches. 1,000,000 integers transferred with zero data corruption. Results on an RTX 3050 Ti Laptop:
    *   **3a - 1-thread consumer:** ~155 ms.
    *   **3b - batch-size log:** batches lock into alternating pairs that sum to a full buffer (e.g. `337 686 337 686 ...`): each batch is exactly what the producer refilled while the GPU read the previous one.
    *   **3c - 32-thread warp consumer (`--warp`):** ~30 ms, ~5x faster, because a warp fetches 128 bytes per PCIe request instead of 4.
*   **Next:** Phase 4 - Measure one-way CPU→GPU latency (ping-pong) and compare transfer methods across message sizes.

---

**Tech Stack**
*   **Language:** C-style C++17
*   **Compiler:** NVCC (CUDA 12.x)
*   **Target Hardware:** Discrete NVIDIA GPUs (Cloud T4 / AWS A100)

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
