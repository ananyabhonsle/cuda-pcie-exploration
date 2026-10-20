#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

#define AB_SAMPLE_RATE 16000
#define AB_MAX_CHUNK_SAMPLES 1280
#define AB_QUEUE_CAPACITY 64
#define AB_STAGING_SLOTS 3
#define AB_ACK_SLOTS 4096
#define AB_ACK_EMPTY 0xFFFFFFFFFFFFFFFFULL

// If the other side makes no progress for this long, give up cleanly.
// Keeps us well under the ~2 s Windows (WDDM) kernel timeout.
#define AB_STALL_TIMEOUT_MS 1000

// Abort with the CUDA error string if a runtime call fails.
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            printf("[FATAL] %s failed at %s:%d: %s\n", #call, __FILE__,   \
                   __LINE__, cudaGetErrorString(err_));                    \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// One audio chunk handed from the CPU to the GPU.
struct Chunk {
    unsigned long long seq;
    unsigned int num_samples;
    unsigned int pad;
    float samples[AB_MAX_CHUNK_SAMPLES];
};

// Deterministic sample in [-1, 1). Integer ops plus a power-of-two division
// only, so the host and the device produce bit-identical results.
__host__ __device__ inline float sample_value(unsigned long long seq, unsigned int i) {
    unsigned int h = (unsigned int)(seq * 2654435761ULL) ^ (i * 40503u);
    return (float)(h & 0xFFFFu) / 32768.0f - 1.0f;
}

inline void fill_chunk(Chunk* c, unsigned long long seq, unsigned int num_samples) {
    c->seq = seq;
    c->num_samples = num_samples;
    c->pad = 0;
    for (unsigned int i = 0; i < num_samples; ++i) c->samples[i] = sample_value(seq, i);
}

// Copies src's header and its first n samples (n clamped to
// AB_MAX_CHUNK_SAMPLES) into dst, e.g. from a producer-local chunk into a
// queue slot. n comes from the caller, never from src's header.
inline void copy_chunk(Chunk* dst, const Chunk* src, unsigned int n) {
    if (n > AB_MAX_CHUNK_SAMPLES) n = AB_MAX_CHUNK_SAMPLES;
    dst->seq = src->seq;
    dst->num_samples = src->num_samples;
    dst->pad = src->pad;
    memcpy(dst->samples, src->samples, (size_t)n * sizeof(float));
}

inline unsigned int samples_for_chunk_ms(int chunk_ms) {
    return (unsigned int)(chunk_ms * (AB_SAMPLE_RATE / 1000));
}

inline double now_us() {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Spin-wait hint. Host x86 only: the device pass sees an empty body.
inline void cpu_relax() {
#if !defined(__CUDA_ARCH__) && (defined(_M_X64) || defined(__x86_64__))
    _mm_pause();
#endif
}

struct LatencyStats {
    double p50, p99, p999, max, mean;
};

// Sorts samples in place. Percentiles use nearest rank: index ceil(p*n)-1.
inline LatencyStats compute_stats(double* samples, size_t n) {
    LatencyStats s = {0, 0, 0, 0, 0};
    if (n == 0) return s;
    std::sort(samples, samples + n);
    double sum = 0.0;
    for (size_t k = 0; k < n; ++k) sum += samples[k];
    auto pct = [&](double p) {
        long long idx = (long long)ceil(p * (double)n) - 1;
        if (idx < 0) idx = 0;
        if (idx > (long long)n - 1) idx = (long long)n - 1;
        return samples[idx];
    };
    s.p50 = pct(0.50);
    s.p99 = pct(0.99);
    s.p999 = pct(0.999);
    s.max = samples[n - 1];
    s.mean = sum / (double)n;
    return s;
}
