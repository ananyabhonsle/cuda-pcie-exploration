#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <chrono>

// 4 MB of floats. Big enough that the timings aren't just noise.
#define N (1 << 20)
#define REPS 5

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

__global__ void gpu_test() {}

// The signal fences are compiler barriers: they stop the compiler moving
// the write/read loops across the clock reads.
static double now_ms() {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    double t = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return t;
}

// Sum of i * 1.5 for i in [0, N). Exact in double: every term is a multiple of 0.5.
static double expected_checksum() {
    return 1.5 * ((double)N * (N - 1) / 2);
}

// Times CPU writes and reads over buf (best of REPS) and verifies the checksum.
static bool bench(const char* name, float* buf) {
    double best_write = 1e30, best_read = 1e30;
    double sum = 0.0;

    for (int rep = 0; rep < REPS; rep++) {
        double t0 = now_ms();
        for (int i = 0; i < N; i++) {
            buf[i] = (float)i * 1.5f;
        }
        double t1 = now_ms();

        sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += buf[i];
        }
        double t2 = now_ms();

        if (t1 - t0 < best_write) best_write = t1 - t0;
        if (t2 - t1 < best_read) best_read = t2 - t1;
    }

    double gb = (double)N * sizeof(float) / 1e9;
    printf("%-18s %10.3f %10.2f %10.3f %10.2f\n",
           name, best_write, gb / (best_write / 1e3), best_read, gb / (best_read / 1e3));

    if (sum != expected_checksum()) {
        printf("[FATAL] %s checksum mismatch. Expected: %f Got: %f\n", name, expected_checksum(), sum);
        return false;
    }
    return true;
}

int main() {
    printf("--- PHASE 1: PHYSICS TEST ---\n");

    gpu_test<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[SUCCESS] GPU is online.\n");

    float* wc_memory = NULL;
    float* pinned_memory = NULL;
    CUDA_CHECK(cudaHostAlloc(&wc_memory, N * sizeof(float), cudaHostAllocWriteCombined));
    CUDA_CHECK(cudaHostAlloc(&pinned_memory, N * sizeof(float), cudaHostAllocDefault));
    float* pageable_memory = (float*)malloc(N * sizeof(float));
    if (pageable_memory == NULL) {
        printf("[FATAL] Failed to allocate pageable memory.\n");
        return EXIT_FAILURE;
    }
    printf("[SUCCESS] Allocated %d floats of WC, pinned and pageable memory.\n", N);

    // WC memory is uncached for CPU reads, so its read column should be far slower.
    printf("\n%-18s %10s %10s %10s %10s\n", "memory", "write ms", "write GB/s", "read ms", "read GB/s");
    bool ok = true;
    ok &= bench("write-combined", wc_memory);
    ok &= bench("pinned", pinned_memory);
    ok &= bench("pageable (malloc)", pageable_memory);
    printf("\n");

    CUDA_CHECK(cudaFreeHost(wc_memory));
    CUDA_CHECK(cudaFreeHost(pinned_memory));
    free(pageable_memory);
    printf("[SUCCESS] Memory freed.\n");

    if (!ok) {
        return EXIT_FAILURE;
    }
    printf("[SUCCESS] All checksums match: %f\n", expected_checksum());
    printf("--- PHASE 1 COMPLETE ---\n");
    return 0;
}
