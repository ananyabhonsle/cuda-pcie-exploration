#include "aetherbridge/harness.h"
#include "check.h"
#include <thread>
#include <vector>

typedef cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> AckRef;

static void ack_store(unsigned long long* slots, unsigned long long seq) {
    AckRef(slots[seq % AB_ACK_SLOTS]).store(seq, cuda::memory_order_release);
}

static void fill_empty(std::vector<unsigned long long>& v) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = AB_ACK_EMPTY;
}

// Producer on the calling thread; fake GPU acks seq once it has been sent.
// block == 1 acks in order, block == 8 acks in reversed blocks of 8.
static void run_episode(unsigned long long total, unsigned long long block,
                        LatencyLog* log, std::vector<unsigned long long>& slots,
                        int* stalled) {
    std::thread observer(ack_observer, log, slots.data(), stalled);
    std::thread gpu([&]() {
        unsigned long long next = 0;
        while (next < total) {
            unsigned long long end = next + block;
            if (end > total) end = total;
            // Wait until the whole block has been sent.
            while (log->sent.load(std::memory_order_acquire) < end) cpu_relax();
            for (unsigned long long s = end; s-- > next;) ack_store(slots.data(), s);
            next = end;
        }
    });
    for (unsigned long long s = 0; s < total; ++s) {
        // Keep the in-flight window under the ring size.
        while (s >= AB_ACK_SLOTS &&
               AckRef(slots[(s - AB_ACK_SLOTS + 1) % AB_ACK_SLOTS])
                       .load(cuda::memory_order_acquire) != s - AB_ACK_SLOTS + 1)
            cpu_relax();
        record_send(log, s);
    }
    gpu.join();
    observer.join();
}

static void test_in_order_acks() {
    LatencyLog log;
    CHECK(latency_log_init(&log, 1000));
    std::vector<unsigned long long> slots(AB_ACK_SLOTS);
    fill_empty(slots);
    int stalled = 0;
    run_episode(1000, 1, &log, slots, &stalled);
    CHECK(stalled == 0);
    bool all = true, order = true;
    for (unsigned long long s = 0; s < 1000; ++s) {
        if (!log.acked[s]) all = false;
        if (log.t_ack[s] < log.t_send[s]) order = false;
    }
    CHECK(all);
    CHECK(order);
    std::vector<double> out(1000);
    CHECK(extract_latencies(&log, 100, out.data()) == 900);
    bool nonneg = true;
    for (size_t i = 0; i < 900; ++i) if (out[i] < 0.0) nonneg = false;
    CHECK(nonneg);
    latency_log_free(&log);
}

static void test_out_of_order_acks() {
    LatencyLog log;
    CHECK(latency_log_init(&log, 1000));
    std::vector<unsigned long long> slots(AB_ACK_SLOTS);
    fill_empty(slots);
    int stalled = 0;
    run_episode(1000, 8, &log, slots, &stalled);
    CHECK(stalled == 0);
    bool all = true;
    for (unsigned long long s = 0; s < 1000; ++s) if (log.acked[s] != 1) all = false;
    CHECK(all);
    latency_log_free(&log);
}

static void test_ack_ring_wrap() {
    LatencyLog log;
    CHECK(latency_log_init(&log, 10000));
    std::vector<unsigned long long> slots(AB_ACK_SLOTS);
    fill_empty(slots);
    int stalled = 0;
    run_episode(10000, 1, &log, slots, &stalled);
    CHECK(stalled == 0);
    bool all = true;
    for (unsigned long long s = 0; s < 10000; ++s) if (!log.acked[s]) all = false;
    CHECK(all);
    latency_log_free(&log);
}

static void test_stall_is_reported() {
    LatencyLog log;
    CHECK(latency_log_init(&log, 10));
    std::vector<unsigned long long> slots(AB_ACK_SLOTS);
    fill_empty(slots);
    int stalled = 0;
    std::thread observer(ack_observer, &log, slots.data(), &stalled);
    for (unsigned long long s = 0; s < 10; ++s) record_send(&log, s);
    for (unsigned long long s = 0; s < 5; ++s) ack_store(slots.data(), s);
    double last_ack = now_us();
    observer.join();
    double elapsed = now_us() - last_ack;
    CHECK(stalled == 1);
    CHECK(elapsed < 1500000.0);
    CHECK(log.acked[4] == 1);
    CHECK(log.acked[5] == 0);
    latency_log_free(&log);
}

static void test_idle_gap_is_not_a_stall() {
    LatencyLog log;
    CHECK(latency_log_init(&log, 2));
    std::vector<unsigned long long> slots(AB_ACK_SLOTS);
    fill_empty(slots);
    int stalled = 0;
    std::thread observer(ack_observer, &log, slots.data(), &stalled);
    record_send(&log, 0);
    ack_store(slots.data(), 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(1300));  // nothing outstanding
    record_send(&log, 1);
    ack_store(slots.data(), 1);
    observer.join();
    CHECK(stalled == 0);
    CHECK(log.acked[0] == 1 && log.acked[1] == 1);
    latency_log_free(&log);
}

static void test_pace_until_waits() {
    double t0 = now_us();
    pace_until(t0 + 2000.0);
    CHECK(now_us() - t0 >= 2000.0);
}

int main() {
    test_in_order_acks();
    test_out_of_order_acks();
    test_ack_ring_wrap();
    test_stall_is_reported();
    test_idle_gap_is_not_a_stall();
    test_pace_until_waits();
    return check_finish();
}
