![C++](https://img.shields.io/badge/C++-17-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)
![Status](https://img.shields.io/badge/Status-Phase_2-yellow.svg)

**CUDA PCIe Exploration**
*Learning low-level CUDA memory management by characterizing host-to-device transfer latency.*

---

**Current Status**
*   **Phase 1 (Complete):** Validated `cudaHostAllocWriteCombined` allocation and benchmarked CPU write/read throughput on write-combined vs. pinned vs. pageable memory.
*   **Phase 2 (Complete):** Validated lock-free SPSC ring buffer with `alignas(64)` cache-line padding. Successfully transferred 1,000,000 integers with zero data corruption.
*   **Next:** Phase 3 - Allocate ring buffer in pinned memory and verify GPU can read from it asynchronously.

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
```
