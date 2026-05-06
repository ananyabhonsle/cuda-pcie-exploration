%%writefile phase2_ringbuffer.cpp
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>

// The buffer size. We use a power of 2 for fast modulo math later if needed, 
// but standard modulo is fine for this test.
#define BUFFER_SIZE 1024
#define ITERATIONS 1000000ULL

// The C-Style struct. No classes. No virtual functions.
// alignas(64) prevents "False Sharing" between CPU cores.
struct RingBuffer {
    alignas(64) std::atomic<size_t> head;
    alignas(64) std::atomic<size_t> tail;
    int data[BUFFER_SIZE];
};

// WORKER A: The Producer
void producer(RingBuffer* buf) {
    for (size_t i = 0; i < ITERATIONS; ++i) {
        size_t current_head = buf->head.load(std::memory_order_relaxed);
        size_t next_head = (current_head + 1) % BUFFER_SIZE;

        // If the next head hits the tail, the buffer is full. Wait.
        while (next_head == buf->tail.load(std::memory_order_acquire)) {
            std::this_thread::yield(); // Yield to OS to prevent hard freeze
            next_head = (buf->head.load(std::memory_order_relaxed) + 1) % BUFFER_SIZE;
        }

        // 1. Write the data to the array
        buf->data[current_head] = static_cast<int>(i);

        // 2. Update the head pointer
        // memory_order_release: Ensures the data write ABOVE happens BEFORE this pointer update
        buf->head.store(next_head, std::memory_order_release);
    }
}

// WORKER B: The Consumer
void consumer(RingBuffer* buf) {
    size_t expected_sum = 0;
    for (size_t i = 0; i < ITERATIONS; ++i) {
        size_t current_tail = buf->tail.load(std::memory_order_relaxed);

        // If head equals tail, the buffer is empty. Wait.
        while (current_tail == buf->head.load(std::memory_order_acquire)) {
            std::this_thread::yield();
            current_tail = buf->tail.load(std::memory_order_relaxed);
        }

        // 1. Read the data from the array
        int value = buf->data[current_tail];
        expected_sum += value;

        // 2. Update the tail pointer
        // memory_order_acquire: Ensures we don't read data BEFORE this pointer update
        size_t next_tail = (current_tail + 1) % BUFFER_SIZE;
        buf->tail.store(next_tail, std::memory_order_acquire);
    }

    // Mathematical proof of data integrity
    // Sum of 0 to 999,999 is (n * (n-1)) / 2
    size_t theoretical_sum = (ITERATIONS * (ITERATIONS - 1)) / 2;
    
    if (expected_sum == theoretical_sum) {
        std::cout << "[SUCCESS] Lock-free ring buffer passed integrity test. Checksum: " << expected_sum << "\n";
    } else {
        std::cout << "[FATAL] Data corruption detected! Expected: " << theoretical_sum << " Got: " << expected_sum << "\n";
    }
}

int main() {
    std::cout << "--- PHASE 2: LOCK-FREE CONCURRENCY TEST ---\n";

    // Initialize buffer
    RingBuffer buf;
    buf.head.store(0, std::memory_order_relaxed);
    buf.tail.store(0, std::memory_order_relaxed);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch threads
    std::thread producer_thread(producer, &buf);
    std::thread consumer_thread(consumer, &buf);

    // Wait for them to finish
    producer_thread.join();
    consumer_thread.join();

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    std::cout << "[SUCCESS] Processed " << ITERATIONS << " chunks in " << duration.count() << " ms.\n";
    std::cout << "--- PHASE 2 COMPLETE ---\n";

    return 0;
}