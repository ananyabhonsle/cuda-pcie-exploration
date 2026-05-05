\## CUDA PCIe Exploration

Learning low-level CUDA memory management by characterizing host-to-device transfer latency.



\### Current Status

Phase 1: Baseline measurement of `cudaHostAllocWriteCombined` vs standard pinned memory on single transfers. 

Next: Expand to streaming buffers and measure sustained bandwidth.



\### Tech Stack

\- C++17, CUDA 12.x

\- Target: NVIDIA T4 / A100 (cloud)

