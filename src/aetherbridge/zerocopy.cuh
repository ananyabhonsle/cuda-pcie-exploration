#pragma once
#include <thread>
#include "aetherbridge/common.h"
#include "aetherbridge/config.h"
#include "aetherbridge/kernels.cuh"
#include "aetherbridge/harness.h"

// V5: lock-free queue + zero-copy. The ring lives in mapped pinned memory and a
// persistent one-warp kernel consumes it directly over PCIe, one episode at a time.
//
// head/tail are monotonic counters: chunk i lives in slots[i % AB_QUEUE_CAPACITY],
// the ring is full when head - tail == AB_QUEUE_CAPACITY. The producer numbers
// chunks with its head counter, so a chunk's seq always equals its ring index.
// Writers: head (producer), tail (kernel lane 0), stop_at (producer),
// running_episode (kernel lane 0). Loads and stores only, never read-modify-write.
struct ZcRing {
    alignas(64) unsigned long long head;
    alignas(64) unsigned long long tail;
    alignas(64) unsigned long long stop_at;          // exit once tail reaches this
    alignas(64) unsigned long long running_episode;  // set by the kernel on entry
    alignas(64) Chunk slots[AB_QUEUE_CAPACITY];
};

// Written by lane 0 when the kernel exits; lives in device memory.
struct ZcKernelResult {
    unsigned long long consumed;
    unsigned long long mismatches;
    int stalled;
};

__device__ inline unsigned long long zc_globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

#define ZC_ACT_WAIT 0
#define ZC_ACT_PROCESS 1
#define ZC_ACT_EXIT 2

// Launch <<<1, 32>>>. Only lane 0 touches head/tail/stop_at/running_episode;
// __syncwarp orders the other lanes' slot reads after lane 0's acquire and
// before its release; __shfl_sync broadcasts every decision so all lanes branch
// identically. Test hook: debug_stop_at >= 0 stops consumption at that seq
// (the kernel keeps looping, so the watchdogs fire).
__global__ void zc_consumer_kernel(ZcRing* ring, unsigned long long* ack_slots, int work_iters,
                                   unsigned long long episode_id, long long debug_stop_at,
                                   ZcKernelResult* out, float* sink) {
    const unsigned int full = 0xFFFFFFFFu;
    const unsigned int lane = threadIdx.x & 31u;
    SysAtomicU64 head(ring->head);
    SysAtomicU64 tail(ring->tail);
    SysAtomicU64 stop_at(ring->stop_at);
    SysAtomicU64 running(ring->running_episode);
    const unsigned long long timeout_ns = (unsigned long long)AB_STALL_TIMEOUT_MS * 1000000ULL;

    // Lane 0 state: t = our tail (we are its only writer), h = last head seen.
    unsigned long long t = 0, h = 0, consumed = 0, mismatches = 0, last_progress = 0;
    int stalled = 0;
    if (lane == 0) {
        // The previous episode's kernel finished before this launch, so its last
        // tail store is visible.
        t = tail.load(cuda::memory_order_relaxed);
        h = t;
        last_progress = zc_globaltimer_ns();
        running.store(episode_id, cuda::memory_order_release);
    }

    for (;;) {
        int action = ZC_ACT_WAIT;
        if (lane == 0) {
            const bool held = debug_stop_at >= 0 && t == (unsigned long long)debug_stop_at;
            // One head load per batch: re-read only when the cached head is used up.
            // memory_order_acquire: slot reads below see what the producer wrote.
            if (!held && t >= h) h = head.load(cuda::memory_order_acquire);
            if (!held && t < h) {
                action = ZC_ACT_PROCESS;
            } else {
                // Compare stop_at with tail, not with a possibly stale head: the
                // producer publishes head before stop_at, so t >= stop_at means
                // every published chunk has been consumed.
                unsigned long long s = stop_at.load(cuda::memory_order_acquire);
                if (t >= s) {
                    action = ZC_ACT_EXIT;
                } else if (zc_globaltimer_ns() - last_progress > timeout_ns) {
                    stalled = 1;
                    action = ZC_ACT_EXIT;
                }
            }
        }
        // Orders the other lanes' slot reads after lane 0's acquire.
        __syncwarp(full);
        action = __shfl_sync(full, action, 0);
        if (action == ZC_ACT_EXIT) break;
        if (action == ZC_ACT_WAIT) continue;

        const unsigned long long idx = __shfl_sync(full, t, 0);
        const Chunk* c = &ring->slots[idx % AB_QUEUE_CAPACITY];
        const unsigned long long seq = c->seq;
        // Ends with a full-mask shuffle, so every lane's slot reads are done.
        const unsigned int bad = process_chunk_warp(c, work_iters, sink);
        __syncwarp(full);
        if (lane == 0) {
            mismatches += bad;
            write_ack(ack_slots, seq);
            t = idx + 1;
            // memory_order_release: the slot reads above finish before the
            // producer may reuse the slot.
            tail.store(t, cuda::memory_order_release);
            ++consumed;
            last_progress = zc_globaltimer_ns();
        }
    }

    if (lane == 0) {
        out->consumed = consumed;
        out->mismatches = mismatches;
        out->stalled = stalled;
    }
}

// Spins until running_episode == id. Returns false after AB_STALL_TIMEOUT_MS.
inline bool zc_wait_for_episode(SysAtomicU64& running, unsigned long long id) {
    const double start = now_us();
    const double stall_us = (double)AB_STALL_TIMEOUT_MS * 1000.0;
    while (running.load(cuda::memory_order_acquire) != id) {
        if (now_us() - start >= stall_us) return false;
        cpu_relax();
    }
    return true;
}

// Spins while the ring is full. Returns false after AB_STALL_TIMEOUT_MS.
inline bool zc_wait_for_space(SysAtomicU64& tail, unsigned long long head) {
    // memory_order_acquire: the kernel's reads of the freed slot are done
    // before we overwrite it.
    if (head - tail.load(cuda::memory_order_acquire) < AB_QUEUE_CAPACITY) return true;
    const double start = now_us();
    const double stall_us = (double)AB_STALL_TIMEOUT_MS * 1000.0;
    while (head - tail.load(cuda::memory_order_acquire) >= AB_QUEUE_CAPACITY) {
        if (now_us() - start >= stall_us) return false;
        cpu_relax();
    }
    return true;
}

// Runs warmup + chunks through the zero-copy transport, in episodes of at most
// cfg.episode_ms of sending. Never hangs on a missing kernel, consumer or
// producer: every wait is bounded by AB_STALL_TIMEOUT_MS and reported as stalled.
inline RunResult run_zerocopy(const RunConfig& cfg) {
    RunResult r;
    memset(&r, 0, sizeof(r));
    const unsigned long long total = cfg.warmup + cfg.chunks;
    const unsigned int num_samples = samples_for_chunk_ms(cfg.chunk_ms);
    const int work_iters = work_iters_for(cfg.work);

    LatencyLog log;
    if (!latency_log_init(&log, total)) {
        r.stalled = 1;
        return r;
    }

    // Fresh ring and ack slots every run: seq restarts at 0, so stale acks
    // from an earlier run would read as false acks.
    ZcRing* ring = NULL;
    unsigned long long* acks = NULL;
    CUDA_CHECK(cudaHostAlloc(&ring, sizeof(ZcRing), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&acks, AB_ACK_SLOTS * sizeof(unsigned long long), cudaHostAllocMapped));
    memset(ring, 0, sizeof(ZcRing));
    for (int i = 0; i < AB_ACK_SLOTS; ++i) acks[i] = AB_ACK_EMPTY;
    ZcRing* d_ring = NULL;
    unsigned long long* d_acks = NULL;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_ring, ring, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_acks, acks, 0));
    ZcKernelResult* d_res = NULL;
    float* d_sink = NULL;
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(ZcKernelResult)));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

    int observer_stalled = 0;
    std::thread observer(ack_observer, &log, acks, &observer_stalled);

    SysAtomicU64 head_a(ring->head);
    SysAtomicU64 tail_a(ring->tail);
    SysAtomicU64 stop_a(ring->stop_at);
    SysAtomicU64 running_a(ring->running_episode);

    unsigned long long head = 0;  // producer's copy; it is the only writer
    unsigned long long episodes = 0, mismatches = 0;  // episodes = kernels launched
    unsigned long long episode_id = 0;                 // running_episode starts at 0
    double active_us = 0.0;
    int stalled = 0;
    const double episode_us = (double)cfg.episode_ms * 1000.0;

    while (head < total && !stalled) {
        ++episode_id;
        stop_a.store(AB_ACK_EMPTY, cuda::memory_order_release);  // no stop yet
        CUDA_CHECK(cudaMemset(d_res, 0, sizeof(ZcKernelResult)));
        if (!cfg.debug_skip_kernel) {
            zc_consumer_kernel<<<1, 32>>>(d_ring, d_acks, work_iters, episode_id,
                                          cfg.debug_kernel_stop_at, d_res, d_sink);
            CUDA_CHECK(cudaGetLastError());
            // Windows (WDDM) queues launches until something flushes them.
            cudaError_t q = cudaStreamQuery(0);
            if (q != cudaSuccess && q != cudaErrorNotReady) CUDA_CHECK(q);
            ++episodes;
        }

        // Kernel start-up must never sit inside a latency sample.
        if (!zc_wait_for_episode(running_a, episode_id)) {
            stalled = 1;
            log.abort.store(1, std::memory_order_release);
            // head == tail here, so a kernel that starts late exits at once.
            stop_a.store(head, cuda::memory_order_release);
            CUDA_CHECK(cudaDeviceSynchronize());
            break;
        }

        bool producer_stopped = false;
        const double episode_start = now_us();
        double first_send = 0.0;
        unsigned long long k = 0;  // chunks sent in this episode
        while (head < total) {
            if (cfg.debug_producer_stop_at >= 0 && head == (unsigned long long)cfg.debug_producer_stop_at) {
                producer_stopped = true;
                break;
            }
            // Stop at the episode boundary; always send at least one chunk. This
            // also keeps the kernel's idle gap under episode_ms (800 ms by
            // default, below its AB_STALL_TIMEOUT_MS watchdog) for any pace_us.
            const double target = cfg.pace_us > 0 ? episode_start + (double)k * cfg.pace_us : now_us();
            if (k > 0 && target - episode_start >= episode_us) break;
            if (cfg.pace_us > 0) pace_until(target);
            if (!zc_wait_for_space(tail_a, head)) {
                stalled = 1;
                break;
            }
            record_send(&log, head);
            if (k == 0) first_send = log.t_send[head];
            Chunk* c = &ring->slots[head % AB_QUEUE_CAPACITY];
            fill_chunk(c, head, num_samples);
            if (cfg.debug_corrupt_seq >= 0 && head == (unsigned long long)cfg.debug_corrupt_seq)
                c->samples[0] += 1.0f;
            // memory_order_release: the chunk is visible to the GPU before head moves.
            head_a.store(head + 1, cuda::memory_order_release);
            ++head;
            ++k;
        }

        // Test hook: a stopped producer never sets stop_at; the kernel's own
        // watchdog must end the episode.
        if (!producer_stopped) stop_a.store(head, cuda::memory_order_release);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (k > 0) active_us += now_us() - first_send;

        ZcKernelResult kr;
        CUDA_CHECK(cudaMemcpy(&kr, d_res, sizeof(kr), cudaMemcpyDeviceToHost));
        mismatches += kr.mismatches;
        if (kr.stalled || producer_stopped) stalled = 1;
        if (stalled) log.abort.store(1, std::memory_order_release);
    }

    observer.join();

    // extract_latencies returns total - warmup even for unacked chunks, whose
    // values are garbage: count acked measured chunks here.
    unsigned long long samples = 0;
    for (unsigned long long s = cfg.warmup; s < total; ++s)
        if (log.acked[s]) ++samples;
    if (observer_stalled || samples != cfg.chunks) stalled = 1;

    if (samples == cfg.chunks) {
        double* lat = (double*)malloc((size_t)cfg.chunks * sizeof(double));
        if (lat) {
            size_t n = extract_latencies(&log, cfg.warmup, lat);
            r.lat = compute_stats(lat, n);
            free(lat);
        } else {
            stalled = 1;
        }
    }

    r.samples = samples;
    r.mismatches = mismatches;
    r.episodes = episodes;
    r.stalled = stalled;
    // head == total on a complete run; on an aborted run it counts what was sent.
    if (active_us > 0.0) {
        r.chunks_per_s = (double)head / (active_us / 1e6);
        r.mb_per_s = r.chunks_per_s * (double)(num_samples * sizeof(float)) / 1e6;
    }

    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFreeHost(acks));
    CUDA_CHECK(cudaFreeHost(ring));
    latency_log_free(&log);
    return r;
}
