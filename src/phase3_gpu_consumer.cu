#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <chrono>

#define BUFFER_SIZE 1024
#define ITERATIONS 1000000ULL

// If the other side makes no progress for this long, give up cleanly.
// Keeps us well under the ~2 s Windows (WDDM) kernel timeout, which would reset the GPU.
#define STALL_TIMEOUT_MS 1000

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

// Same layout as Phase 2, but head/tail are plain size_t: the CPU and the GPU
// both access them through cuda::atomic_ref, which works across PCIe.
// Only atomic loads and stores are used. Read-modify-write atomics (fetch_add etc.)
// on host memory need hostNativeAtomicSupported, which consumer GPUs usually lack.
struct RingBuffer {
    alignas(64) size_t head;
    alignas(64) size_t tail;
    alignas(64) int data[BUFFER_SIZE];
};

typedef cuda::atomic_ref<size_t, cuda::thread_scope_system> SysAtomic;

// Written by the GPU, copied back to the CPU after the kernel finishes.
struct Result {
    unsigned long long sum;
    unsigned long long consumed;
    unsigned long long batches;
    int timed_out;
};

__device__ unsigned long long globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// WORKER A: The Producer (CPU thread, same loop as Phase 2)
void producer(RingBuffer* buf, bool* ok) {
    SysAtomic head(buf->head);
    SysAtomic tail(buf->tail);

    for (size_t i = 0; i < ITERATIONS; ++i) {
        size_t current_head = head.load(cuda::std::memory_order_relaxed);
        size_t next_head = (current_head + 1) % BUFFER_SIZE;

        // If the next head hits the tail, the buffer is full. Wait for the GPU.
        if (next_head == tail.load(cuda::std::memory_order_acquire)) {
            auto wait_start = std::chrono::steady_clock::now();
            while (next_head == tail.load(cuda::std::memory_order_acquire)) {
                std::chrono::duration<double, std::milli> waited = std::chrono::steady_clock::now() - wait_start;
                if (waited.count() > STALL_TIMEOUT_MS) {
                    *ok = false;
                    return;
                }
                std::this_thread::yield();
            }
        }

        // 1. Write the data to the array
        buf->data[current_head] = static_cast<int>(i);

        // 2. Update the head pointer
        // memory_order_release: Ensures the data write ABOVE is visible to the GPU BEFORE this pointer update
        head.store(next_head, cuda::std::memory_order_release);
    }
    *ok = true;
}

// WORKER B: The Consumer (GPU, reading host memory over PCIe)
// Launched with 1 thread (3a) or one warp of 32 threads (3c, --warp).
// Lane 0 is the only thread that touches head/tail; all lanes share the data reads.
// batch_log[i] records how many items batch i consumed. It lives in GPU memory,
// so logging costs one fast local store per batch, not a PCIe write.
__global__ void consumer(RingBuffer* buf, Result* out, unsigned int* batch_log) {
    const unsigned int lane = threadIdx.x;
    const unsigned int lanes = blockDim.x;
    const unsigned int mask = (lanes == 32) ? 0xffffffffu : 1u;
    SysAtomic head(buf->head);
    SysAtomic tail(buf->tail);

    unsigned long long sum = 0, consumed = 0, batches = 0;
    size_t current_tail = tail.load(cuda::std::memory_order_relaxed);
    unsigned long long last_progress = globaltimer_ns();

    while (consumed < ITERATIONS) {
        // One PCIe round trip (~1 us) tells us how many items are ready.
        // memory_order_acquire: Ensures the data reads BELOW see what the producer wrote
        size_t current_head = 0;
        if (lane == 0) {
            current_head = head.load(cuda::std::memory_order_acquire);
        }
        // __syncwarp orders the other lanes' data reads after lane 0's acquire
        __syncwarp(mask);
        current_head = __shfl_sync(mask, current_head, 0);

        // If head equals tail, the buffer is empty. Wait.
        if (current_head == current_tail) {
            int stalled = (lane == 0) && (globaltimer_ns() - last_progress > STALL_TIMEOUT_MS * 1000000ULL);
            if (__shfl_sync(mask, stalled, 0)) {
                if (lane == 0) out->timed_out = 1;
                break;
            }
            continue;
        }

        // Consume the whole batch. Lane i reads slots i, i+32, i+64, ... so one warp
        // fetches 32 consecutive ints (128 bytes) per PCIe request instead of 4 bytes.
        unsigned int batch_items = (unsigned int)((current_head + BUFFER_SIZE - current_tail) % BUFFER_SIZE);
        for (unsigned int i = lane; i < batch_items; i += lanes) {
            sum += buf->data[(current_tail + i) % BUFFER_SIZE];
        }
        current_tail = (current_tail + batch_items) % BUFFER_SIZE;
        consumed += batch_items;

        // Every lane's reads must finish before lane 0 frees the slots
        __syncwarp(mask);
        if (lane == 0) {
            // memory_order_release: Ensures the data reads ABOVE finish BEFORE the producer can reuse these slots
            tail.store(current_tail, cuda::std::memory_order_release);
            batch_log[batches] = batch_items;
            last_progress = globaltimer_ns();
        }
        batches++;
    }

    // Combine the per-lane partial sums
    for (unsigned int offset = lanes / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    if (lane == 0) {
        out->sum = sum;
        out->consumed = consumed;
        out->batches = batches;
    }
}

static int compare_uint(const void* a, const void* b) {
    unsigned int x = *(const unsigned int*)a, y = *(const unsigned int*)b;
    return (x > y) - (x < y);
}

// 3b: how big were the batches? Prints a histogram plus a sample of consecutive
// batches, to show whether big and small batches alternate.
static void print_batch_report(const unsigned int* sizes, unsigned long long n) {
    if (n == 0) return;

    // Power-of-two buckets: 1, 2-3, 4-7, ... 512-1023 (a batch is at most BUFFER_SIZE - 1)
    const int NUM_BUCKETS = 10;
    unsigned long long bucket_batches[NUM_BUCKETS] = {0};
    unsigned long long bucket_items[NUM_BUCKETS] = {0};
    unsigned long long big_small_pairs = 0;
    for (unsigned long long i = 0; i < n; i++) {
        int b = 0;
        while ((2u << b) <= sizes[i] && b < NUM_BUCKETS - 1) b++;
        bucket_batches[b]++;
        bucket_items[b] += sizes[i];
        if (i > 0) {
            unsigned int lo = sizes[i] < sizes[i - 1] ? sizes[i] : sizes[i - 1];
            unsigned int hi = sizes[i] < sizes[i - 1] ? sizes[i - 1] : sizes[i];
            if (hi >= 4 * lo) big_small_pairs++;
        }
    }

    unsigned int* sorted = (unsigned int*)malloc(n * sizeof(unsigned int));
    if (sorted == NULL) return;
    memcpy(sorted, sizes, n * sizeof(unsigned int));
    qsort(sorted, n, sizeof(unsigned int), compare_uint);
    printf("\nBatch sizes over %llu batches: min %u, median %u, max %u\n",
           n, sorted[0], sorted[n / 2], sorted[n - 1]);
    free(sorted);

    printf("  %-10s %10s %13s %11s\n", "size", "batches", "% of batches", "% of items");
    for (int b = 0; b < NUM_BUCKETS; b++) {
        char label[16];
        unsigned int lo = 1u << b, hi = (2u << b) - 1;
        if (lo == hi) snprintf(label, sizeof(label), "%u", lo);
        else snprintf(label, sizeof(label), "%u-%u", lo, hi);
        printf("  %-10s %10llu %12.1f%% %10.1f%%\n", label, bucket_batches[b],
               100.0 * bucket_batches[b] / n, 100.0 * bucket_items[b] / ITERATIONS);
    }
    if (n > 1) {
        printf("Consecutive batches where one is 4x+ the other: %.1f%%\n", 100.0 * big_small_pairs / (n - 1));
    }

    unsigned long long start = n / 2 > 8 ? n / 2 - 8 : 0;
    unsigned long long end = start + 16 < n ? start + 16 : n;
    printf("Sample from the middle of the run (batches %llu-%llu):", start, end - 1);
    for (unsigned long long i = start; i < end; i++) printf(" %u", sizes[i]);
    printf("\n\n");
}

int main(int argc, char** argv) {
    // 3a: one consumer thread (default). 3c: --warp, one warp of 32 consumer threads.
    int lanes = 1;
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--warp") == 0) {
            lanes = 32;
        } else {
            printf("Usage: %s [--warp]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }
    printf("--- PHASE 3: GPU CONSUMER TEST (%s) ---\n",
           lanes == 32 ? "3c: 32-thread warp consumer" : "3a: 1-thread consumer");

    // Mapped pinned memory: lives in host RAM, but the GPU can read/write it directly.
    RingBuffer* buf = NULL;
    CUDA_CHECK(cudaHostAlloc(&buf, sizeof(RingBuffer), cudaHostAllocMapped));
    buf->head = 0;
    buf->tail = 0;
    RingBuffer* d_buf = NULL;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_buf, buf, 0));

    Result* d_result = NULL;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(Result)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(Result)));

    // Worst case is one item per batch, so room for ITERATIONS entries (4 MB)
    unsigned int* d_batch_log = NULL;
    CUDA_CHECK(cudaMalloc(&d_batch_log, ITERATIONS * sizeof(unsigned int)));
    printf("[SUCCESS] Ring buffer allocated in mapped pinned memory.\n");

    // Start timing
    auto start_time = std::chrono::steady_clock::now();

    // Launch the GPU consumer first, so it is waiting when data arrives
    consumer<<<1, lanes>>>(d_buf, d_result, d_batch_log);
    CUDA_CHECK(cudaGetLastError());

    // On Windows (WDDM) launches are queued until something flushes them. Without this
    // the producer could fill the buffer and wait for a kernel that hasn't started.
    cudaError_t q = cudaStreamQuery(0);
    if (q != cudaSuccess && q != cudaErrorNotReady) {
        CUDA_CHECK(q);
    }

    // Launch the CPU producer
    bool producer_ok = false;
    std::thread producer_thread(producer, buf, &producer_ok);

    // Wait for both to finish
    producer_thread.join();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Stop timing
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    Result result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(Result), cudaMemcpyDeviceToHost));
    unsigned int* batch_log = (unsigned int*)malloc(ITERATIONS * sizeof(unsigned int));
    if (batch_log == NULL) {
        printf("[FATAL] Failed to allocate batch log.\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(batch_log, d_batch_log, result.batches * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_batch_log));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFreeHost(buf));

    if (!producer_ok) {
        printf("[FATAL] Producer stalled: GPU stopped draining the buffer for %d ms.\n", STALL_TIMEOUT_MS);
    }
    if (result.timed_out) {
        printf("[FATAL] GPU consumer stalled: no new data for %d ms (got %llu of %llu items).\n",
               STALL_TIMEOUT_MS, result.consumed, ITERATIONS);
    }
    if (!producer_ok || result.timed_out) {
        return EXIT_FAILURE;
    }

    // Mathematical proof of data integrity
    // Sum of 0 to 999,999 is (n * (n-1)) / 2
    unsigned long long theoretical_sum = (ITERATIONS * (ITERATIONS - 1)) / 2;
    if (result.consumed != ITERATIONS || result.sum != theoretical_sum) {
        printf("[FATAL] Data corruption detected! Expected: %llu items, sum %llu. Got: %llu items, sum %llu\n",
               ITERATIONS, theoretical_sum, result.consumed, result.sum);
        return EXIT_FAILURE;
    }

    printf("[SUCCESS] GPU consumed %llu items. Checksum: %llu\n", result.consumed, result.sum);
    printf("[SUCCESS] Processed in %.2f ms (%llu batches, avg %.1f items/batch, %.2f us/batch)\n",
           duration.count(), result.batches, (double)result.consumed / result.batches,
           duration.count() * 1e3 / result.batches);
    print_batch_report(batch_log, result.batches);
    free(batch_log);
    printf("--- PHASE 3 COMPLETE ---\n");
    return 0;
}
