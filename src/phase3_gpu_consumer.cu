#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdio.h>
#include <stdlib.h>
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

// WORKER B: The Consumer (one GPU thread, reading host memory over PCIe)
__global__ void consumer(RingBuffer* buf, Result* out) {
    SysAtomic head(buf->head);
    SysAtomic tail(buf->tail);

    unsigned long long sum = 0, consumed = 0, batches = 0;
    size_t current_tail = tail.load(cuda::std::memory_order_relaxed);
    unsigned long long last_progress = globaltimer_ns();

    while (consumed < ITERATIONS) {
        // One PCIe round trip (~1 us) tells us how many items are ready.
        // memory_order_acquire: Ensures the data reads BELOW see what the producer wrote
        size_t current_head = head.load(cuda::std::memory_order_acquire);

        // If head equals tail, the buffer is empty. Wait.
        if (current_head == current_tail) {
            if (globaltimer_ns() - last_progress > STALL_TIMEOUT_MS * 1000000ULL) {
                out->timed_out = 1;
                break;
            }
            continue;
        }

        // Consume the whole batch, then publish the new tail once
        while (current_tail != current_head) {
            sum += buf->data[current_tail];
            current_tail = (current_tail + 1) % BUFFER_SIZE;
            consumed++;
        }

        // memory_order_release: Ensures the data reads ABOVE finish BEFORE the producer can reuse these slots
        tail.store(current_tail, cuda::std::memory_order_release);
        batches++;
        last_progress = globaltimer_ns();
    }

    out->sum = sum;
    out->consumed = consumed;
    out->batches = batches;
}

int main() {
    printf("--- PHASE 3: GPU CONSUMER TEST ---\n");

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
    printf("[SUCCESS] Ring buffer allocated in mapped pinned memory.\n");

    // Start timing
    auto start_time = std::chrono::steady_clock::now();

    // Launch the GPU consumer first, so it is waiting when data arrives
    consumer<<<1, 1>>>(d_buf, d_result);
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
    printf("--- PHASE 3 COMPLETE ---\n");
    return 0;
}
