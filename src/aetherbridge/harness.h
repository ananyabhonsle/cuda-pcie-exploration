#pragma once
#include "common.h"
#include <atomic>
#include <cuda/atomic>

// Per-run latency record. t_send is written by the producer before it
// publishes `sent`; t_ack and acked are written only by the observer thread.
struct LatencyLog {
    unsigned long long total;
    double* t_send;
    double* t_ack;
    unsigned char* acked;
    std::atomic<unsigned long long> sent;  // chunks published so far
    std::atomic<int> abort;                // set to stop the observer early
};

inline bool latency_log_init(LatencyLog* log, unsigned long long total) {
    log->total = total;
    log->t_send = (double*)calloc((size_t)total, sizeof(double));
    log->t_ack = (double*)calloc((size_t)total, sizeof(double));
    log->acked = (unsigned char*)calloc((size_t)total, 1);
    log->sent.store(0, std::memory_order_relaxed);
    log->abort.store(0, std::memory_order_relaxed);
    if (!log->t_send || !log->t_ack || !log->acked) {
        free(log->t_send); free(log->t_ack); free(log->acked);
        log->t_send = log->t_ack = NULL;
        log->acked = NULL;
        return false;
    }
    return true;
}

inline void latency_log_free(LatencyLog* log) {
    free(log->t_send); free(log->t_ack); free(log->acked);
    log->t_send = log->t_ack = NULL;
    log->acked = NULL;
}

// Call immediately before publishing chunk `seq`.
inline void record_send(LatencyLog* log, unsigned long long seq) {
    log->t_send[seq] = now_us();
    log->sent.store(seq + 1, std::memory_order_release);
}

// Observer thread body. Scans every in-flight ack slot (acks may complete out
// of order). Sets *stalled_out = 1 if chunks are outstanding and no new ack
// arrives for AB_STALL_TIMEOUT_MS. Idle time with nothing outstanding never
// counts. Returns when all acked, on abort, or on stall.
inline void ack_observer(LatencyLog* log, unsigned long long* ack_slots, int* stalled_out) {
    *stalled_out = 0;
    unsigned long long lowest = 0;
    double last_progress = now_us();
    const double stall_us = (double)AB_STALL_TIMEOUT_MS * 1000.0;
    while (lowest < log->total) {
        if (log->abort.load(std::memory_order_acquire)) return;
        unsigned long long sent = log->sent.load(std::memory_order_acquire);
        bool progressed = false;
        for (unsigned long long seq = lowest; seq < sent; ++seq) {
            if (log->acked[seq]) continue;
            cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> slot(
                ack_slots[seq % AB_ACK_SLOTS]);
            unsigned long long v = slot.load(cuda::memory_order_acquire);
            // v == seq, or a later seq that reused this slot (ack was missed).
            if (v != AB_ACK_EMPTY && v >= seq && (v - seq) % AB_ACK_SLOTS == 0) {
                log->t_ack[seq] = now_us();
                log->acked[seq] = 1;
                progressed = true;
            }
        }
        while (lowest < log->total && log->acked[lowest]) ++lowest;
        double now = now_us();
        if (progressed || sent <= lowest) {
            last_progress = now;  // idle gaps never count as a stall
        } else if (now - last_progress >= stall_us) {
            *stalled_out = 1;
            return;
        }
        cpu_relax();
    }
}

inline void pace_until(double deadline_us) {
    while (now_us() < deadline_us) cpu_relax();
}

// Writes t_ack - t_send for seq in [warmup, total); returns the count.
inline size_t extract_latencies(const LatencyLog* log, unsigned long long warmup, double* out) {
    size_t n = 0;
    for (unsigned long long s = warmup; s < log->total; ++s)
        out[n++] = log->t_ack[s] - log->t_send[s];
    return n;
}
