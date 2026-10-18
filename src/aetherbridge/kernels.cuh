#pragma once
#include <cuda/atomic>
#include "aetherbridge/common.h"
#include "aetherbridge/config.h"

// Stand-in GPU work per chunk (replaces the real audio model).
// Measured with tests/calibrate_work.cu, 20 ms chunks (320 samples), one warp,
// chunks back to back in device memory. Work scales linearly with chunk length.
//   GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU (sm_86), run-to-run noise ~5-10%
//   AB_WORK_ITERS_LIGHT -> ~10 us/chunk (measured 9.4-11.1)
//   AB_WORK_ITERS_HEAVY -> ~200 us/chunk (measured 193-206)
#define AB_WORK_ITERS_LIGHT 170
#define AB_WORK_ITERS_HEAVY 6500

__host__ __device__ inline int work_iters_for(WorkLevel w) {
    switch (w) {
        case WORK_LIGHT: return AB_WORK_ITERS_LIGHT;
        case WORK_HEAVY: return AB_WORK_ITERS_HEAVY;
        default: return 0;
    }
}

typedef cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> SysAtomicU64;

// Called by all 32 lanes of one warp. Lane l handles samples l, l+32, ...
// Verifies samples [0, n) against sample_value(seq, i), then burns work_iters
// dependent fmaf steps per sample. seq and n are what the caller expects, not
// the chunk's own header, so a corrupt header cannot steer the reads; n is
// clamped to AB_MAX_CHUNK_SAMPLES. Returns the mismatch count on every lane.
__device__ inline unsigned int process_chunk_warp(const Chunk* c, unsigned long long seq, unsigned int n,
                                                  int work_iters, float* sink) {
    const unsigned int lane = threadIdx.x & 31u;
    if (n > AB_MAX_CHUNK_SAMPLES) n = AB_MAX_CHUNK_SAMPLES;
    unsigned int bad = 0;
    float acc = 0.5f;
    for (unsigned int i = lane; i < n; i += 32) {
        float v = c->samples[i];
        if (v != sample_value(seq, i)) ++bad;
        for (int k = 0; k < work_iters; ++k) acc = fmaf(acc, 0.999f, v);
    }
    // acc stays bounded (|acc| < ~1000), so this sentinel is never hit.
    if (acc == 1.0e30f) *sink = acc;
    for (int off = 16; off > 0; off >>= 1) bad += __shfl_xor_sync(0xFFFFFFFFu, bad, off);
    return bad;
}

// Lane 0 only: system-scope release store of seq into its ack slot.
__device__ inline void write_ack(unsigned long long* ack_slots, unsigned long long seq) {
    SysAtomicU64 a(ack_slots[seq % AB_ACK_SLOTS]);
    a.store(seq, cuda::memory_order_release);
}
