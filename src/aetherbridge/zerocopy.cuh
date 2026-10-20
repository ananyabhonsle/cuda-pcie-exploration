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
// mismatches counts wrong samples plus one per chunk whose header is wrong
// (seq != its ring index, or num_samples != the run's chunk size); a header
// error is an integrity error like a bad sample, so it shares the CSV column.
// work_ns sums lane 0's globaltimer time inside process_chunk_warp over the
// work_chunks chunks with seq >= measure_from (warm-up excluded).
struct ZcKernelResult {
    unsigned long long consumed;
    unsigned long long mismatches;
    unsigned long long work_ns;
    unsigned long long work_chunks;
    int stalled;
};

__device__ inline unsigned long long zc_globaltimer_ns() {
    unsigned long long t;
    // "memory": the compiler may not move loads or stores across the read, so
    // the timestamps around process_chunk_warp bracket its slot reads.
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) : : "memory");
    return t;
}

#define ZC_ACT_WAIT 0
#define ZC_ACT_PROCESS 1
#define ZC_ACT_EXIT 2

// An empty poll costs one PCIe round trip (the head load). stop_at and the
// watchdogs are checked only on every ZC_STOP_CHECK_POLLS-th consecutive empty
// poll, so a poll is not two dependent round trips; this delays the exit at an
// episode end by at most that many polls.
#define ZC_STOP_CHECK_POLLS 16

// A kernel that has run this long exits with stalled = 1, so its lifetime stays
// under the ~2 s Windows (WDDM) limit even if a stall starts late in an
// episode (episode_ms + AB_STALL_TIMEOUT_MS alone could reach 1.8 s).
#define ZC_MAX_KERNEL_AGE_MS 1500

// Launch <<<1, 32>>>. Only lane 0 touches head/tail/stop_at/running_episode;
// __syncwarp orders the other lanes' slot reads after lane 0's acquire and
// before its release; __shfl_sync broadcasts every decision so all lanes branch
// identically. num_samples is the run's chunk size (<= AB_MAX_CHUNK_SAMPLES).
// Test hook: debug_stop_at >= 0 stops consumption at that seq (the kernel
// keeps looping, so the watchdogs fire). max_age_ms is ZC_MAX_KERNEL_AGE_MS
// except in tests. Work time is summed for chunks with seq >= measure_from.
__global__ void zc_consumer_kernel(ZcRing* ring, unsigned long long* ack_slots, unsigned int num_samples,
                                   int work_iters, unsigned long long episode_id, long long debug_stop_at,
                                   int max_age_ms, unsigned long long measure_from, ZcKernelResult* out,
                                   float* sink) {
    const unsigned int full = 0xFFFFFFFFu;
    const unsigned int lane = threadIdx.x & 31u;
    SysAtomicU64 head(ring->head);
    SysAtomicU64 tail(ring->tail);
    SysAtomicU64 stop_at(ring->stop_at);
    SysAtomicU64 running(ring->running_episode);
    const unsigned long long timeout_ns = (unsigned long long)AB_STALL_TIMEOUT_MS * 1000000ULL;
    const unsigned long long max_age_ns = (unsigned long long)max_age_ms * 1000000ULL;

    // Lane 0 state: t = our tail (we are its only writer), h = last head seen,
    // idle = consecutive empty polls.
    unsigned long long t = 0, h = 0, consumed = 0, mismatches = 0, work_ns = 0, work_chunks = 0;
    unsigned long long entry_ns = 0, last_progress = 0;
    unsigned int idle = 0;
    int stalled = 0;
    if (lane == 0) {
        // The previous episode's kernel finished before this launch, so its last
        // tail store is visible.
        t = tail.load(cuda::memory_order_relaxed);
        h = t;
        entry_ns = zc_globaltimer_ns();
        last_progress = entry_ns;
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
                idle = 0;
                // Age cap while busy; last_progress is when the last chunk finished.
                if (last_progress - entry_ns > max_age_ns) {
                    stalled = 1;
                    action = ZC_ACT_EXIT;
                } else {
                    action = ZC_ACT_PROCESS;
                }
            } else if (++idle % ZC_STOP_CHECK_POLLS == 0) {
                // Compare stop_at with tail, not with a possibly stale head: the
                // producer publishes head before stop_at, so t >= stop_at means
                // every published chunk has been consumed.
                const unsigned long long s = stop_at.load(cuda::memory_order_acquire);
                const unsigned long long now = zc_globaltimer_ns();
                if (t >= s) {
                    action = ZC_ACT_EXIT;
                } else if (now - last_progress > timeout_ns || now - entry_ns > max_age_ns) {
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
        // Integrity: the header must name this ring index and the run's chunk
        // size. It is read here but judged after processing, so its PCIe read
        // overlaps the sample reads; the samples are checked against idx and
        // num_samples, never against the header.
        const unsigned long long hdr_seq = c->seq;
        const unsigned int hdr_n = c->num_samples;
        // Read by every lane (only lane 0's value is used) so the warp does not
        // diverge here and no lane starts its sample reads before the stamp.
        const unsigned long long w0 = zc_globaltimer_ns();
        // Ends with a full-mask shuffle, so every lane's slot reads are done.
        const unsigned int bad = process_chunk_warp(c, idx, num_samples, work_iters, sink);
        __syncwarp(full);
        if (lane == 0) {
            // In-situ work time: the shuffle above waited for every lane, so
            // w1 - w0 covers the whole warp's sample reads and compute.
            const unsigned long long w1 = zc_globaltimer_ns();
            if (idx >= measure_from) {
                work_ns += w1 - w0;
                ++work_chunks;
            }
            mismatches += bad;
            if (hdr_seq != idx || hdr_n != num_samples) ++mismatches;
            // Ack the ring index: acks stay in order and exactly once even
            // when the header is corrupt (the corruption is counted above).
            write_ack(ack_slots, idx);
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
        out->work_ns = work_ns;
        out->work_chunks = work_chunks;
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
    const int max_age_ms = cfg.debug_kernel_max_age_ms > 0 ? cfg.debug_kernel_max_age_ms : ZC_MAX_KERNEL_AGE_MS;

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
    Chunk local;                  // producer-local chunk, generated before t_send
    unsigned long long episodes = 0, mismatches = 0;  // episodes = kernels launched
    unsigned long long consumed = 0;                   // summed over episodes
    unsigned long long work_ns = 0, work_chunks = 0;   // summed over episodes
    unsigned long long episode_id = 0;                 // running_episode starts at 0
    double active_us = 0.0;
    int stalled = 0;
    const double episode_us = (double)cfg.episode_ms * 1000.0;

    while (head < total && !stalled) {
        ++episode_id;
        stop_a.store(AB_ACK_EMPTY, cuda::memory_order_release);  // no stop yet
        CUDA_CHECK(cudaMemset(d_res, 0, sizeof(ZcKernelResult)));
        if (!cfg.debug_skip_kernel) {
            zc_consumer_kernel<<<1, 32>>>(d_ring, d_acks, num_samples, work_iters, episode_id,
                                          cfg.debug_kernel_stop_at, max_age_ms, cfg.warmup, d_res, d_sink);
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
            // Generate the chunk (and apply test hooks) before the sample
            // starts: see the sample definition next to record_send.
            fill_chunk(&local, head, num_samples);
            if (cfg.debug_corrupt_seq >= 0 && head == (unsigned long long)cfg.debug_corrupt_seq)
                local.samples[0] += 1.0f;
            if (cfg.debug_bad_seq_at >= 0 && head == (unsigned long long)cfg.debug_bad_seq_at)
                local.seq = head + 1;
            if (cfg.debug_bad_count_at >= 0 && head == (unsigned long long)cfg.debug_bad_count_at)
                local.num_samples = 0xFFFFFFFFu;
            if (cfg.debug_zero_slot_at >= 0 && head == (unsigned long long)cfg.debug_zero_slot_at)
                memset(&local, 0, sizeof(Chunk));
            if (cfg.pace_us > 0) pace_until(target);
            if (!zc_wait_for_space(tail_a, head)) {
                stalled = 1;
                break;
            }
            record_send(&log, head);
            if (k == 0) first_send = log.t_send[head];
            // Header plus num_samples floats; the count is the run's, never the
            // local header's (a test hook may have corrupted it).
            copy_chunk(&ring->slots[head % AB_QUEUE_CAPACITY], &local, num_samples);
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
        consumed += kr.consumed;
        work_ns += kr.work_ns;
        work_chunks += kr.work_chunks;
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
    // Exactly once: the kernels must have consumed every published chunk, no
    // more and no fewer. A difference means the consumer did not finish the
    // work it was given, so it is reported as stalled (mismatches keeps
    // counting only bad chunks the kernel actually saw).
    if (consumed != head) stalled = 1;

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
    r.consumed = consumed;
    if (work_chunks > 0) r.gpu_work_us = (double)work_ns / (double)work_chunks / 1000.0;
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
