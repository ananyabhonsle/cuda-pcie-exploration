// AetherBridge bench: run one configuration, print a summary, optionally append a CSV row.
#include <cstdio>
#include "aetherbridge/common.h"
#include "aetherbridge/config.h"
#include "aetherbridge/zerocopy.cuh"

static void print_usage() {
    printf("usage: bench [--queue lockfree|mutex] [--wait spin|block] [--transport sync|async|zerocopy]\n"
           "             [--chunk-ms 10|20|40|80] [--work none|light|heavy] [--pace-us N]\n"
           "             [--chunks N] [--warmup N] [--csv PATH]\n");
}

int main(int argc, char** argv) {
    RunConfig cfg = default_config();
    char err[256];
    if (!parse_args(argc, argv, &cfg, err, sizeof(err))) {
        printf("[FATAL] %s\n", err);
        print_usage();
        return 1;
    }
    if (cfg.transport == TRANSPORT_SYNC || cfg.transport == TRANSPORT_ASYNC) {
        printf("[FATAL] --transport %s is implemented in Phase 5\n", transport_str(cfg.transport));
        return 1;
    }
    if (cfg.queue == QUEUE_MUTEX) {
        printf("[FATAL] --queue mutex is implemented in Phase 6\n");
        return 1;
    }

    RunResult r = run_zerocopy(cfg);
    unsigned chunk_bytes = samples_for_chunk_ms(cfg.chunk_ms) * 4u;

    printf("--- AETHERBRIDGE BENCH: V5 (lock-free queue, zero-copy) ---\n");
    if (cfg.pace_us > 0)
        printf("chunk %d ms (%u B), work %s, paced every %d us, %llu chunks (+%llu warmup), %llu episodes\n",
               cfg.chunk_ms, chunk_bytes, work_str(cfg.work), cfg.pace_us, cfg.chunks, cfg.warmup, r.episodes);
    else
        printf("chunk %d ms (%u B), work %s, unpaced, %llu chunks (+%llu warmup), %llu episodes\n",
               cfg.chunk_ms, chunk_bytes, work_str(cfg.work), cfg.chunks, cfg.warmup, r.episodes);
    printf("latency us: p50 %.2f  p99 %.2f  p99.9 %.2f  max %.2f  mean %.2f\n",
           r.lat.p50, r.lat.p99, r.lat.p999, r.lat.max, r.lat.mean);
    printf("gpu work: %.2f us/chunk (mean, measured in the kernel)\n", r.gpu_work_us);
    printf("throughput: %.0f chunks/s, %.1f MB/s\n", r.chunks_per_s, r.mb_per_s);

    bool ok = (r.mismatches == 0 && !r.stalled);
    if (r.mismatches != 0) printf("[FATAL] %llu mismatches\n", r.mismatches);
    if (r.stalled) printf("[FATAL] run stalled\n");
    if (ok) printf("[SUCCESS] 0 mismatches, no stalls\n");

    if (cfg.csv_path[0]) {
        PlatformInfo p = get_platform_info();
        if (!append_csv_row(cfg.csv_path, cfg, p, r)) {
            printf("[FATAL] could not write CSV '%s'\n", cfg.csv_path);
            return 1;
        }
    }
    return ok ? 0 : 1;
}
