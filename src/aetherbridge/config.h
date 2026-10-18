#pragma once
#include "aetherbridge/common.h"

enum QueueKind { QUEUE_LOCKFREE, QUEUE_MUTEX };
enum WaitKind { WAIT_SPIN, WAIT_BLOCK };
enum Transport { TRANSPORT_SYNC, TRANSPORT_ASYNC, TRANSPORT_ZEROCOPY };
enum WorkLevel { WORK_NONE, WORK_LIGHT, WORK_HEAVY };

struct RunConfig {
    QueueKind queue;
    WaitKind wait;
    Transport transport;
    int chunk_ms;
    WorkLevel work;
    int pace_us;
    unsigned long long chunks;
    unsigned long long warmup;
    int episode_ms;
    char csv_path[512];
    // Test hooks: -1 / 0 = off. Never set by parse_args. The *_at hooks act on
    // the chunk with that seq: a wrong header seq, a num_samples of 0xFFFFFFFF,
    // or an all-zero slot.
    long long debug_corrupt_seq;
    long long debug_kernel_stop_at;
    long long debug_producer_stop_at;
    int debug_skip_kernel;
    int debug_kernel_max_age_ms;  // 0 = ZC_MAX_KERNEL_AGE_MS
    long long debug_bad_seq_at;
    long long debug_bad_count_at;
    long long debug_zero_slot_at;
};

inline RunConfig default_config() {
    RunConfig c;
    memset(&c, 0, sizeof(c));
    c.queue = QUEUE_LOCKFREE;
    c.wait = WAIT_SPIN;
    c.transport = TRANSPORT_ZEROCOPY;
    c.chunk_ms = 20;
    c.work = WORK_NONE;
    c.pace_us = 1000;
    c.chunks = 20000;
    c.warmup = 500;
    c.episode_ms = 800;
    c.csv_path[0] = 0;
    c.debug_corrupt_seq = -1;
    c.debug_kernel_stop_at = -1;
    c.debug_producer_stop_at = -1;
    c.debug_skip_kernel = 0;
    c.debug_kernel_max_age_ms = 0;
    c.debug_bad_seq_at = -1;
    c.debug_bad_count_at = -1;
    c.debug_zero_slot_at = -1;
    return c;
}

inline const char* queue_str(QueueKind q) { return q == QUEUE_MUTEX ? "mutex" : "lockfree"; }
inline const char* wait_str(WaitKind w) { return w == WAIT_BLOCK ? "block" : "spin"; }
inline const char* transport_str(Transport t) {
    return t == TRANSPORT_SYNC ? "sync" : (t == TRANSPORT_ASYNC ? "async" : "zerocopy");
}
inline const char* work_str(WorkLevel w) {
    return w == WORK_HEAVY ? "heavy" : (w == WORK_LIGHT ? "light" : "none");
}

inline bool ab_parse_ll(const char* s, long long* out) {
    if (!s || !*s) return false;
    char* end = nullptr;
    long long v = strtoll(s, &end, 10);
    if (*end != 0) return false;
    *out = v;
    return true;
}

inline bool parse_args(int argc, char** argv, RunConfig* cfg, char* err, size_t err_len) {
    err[0] = 0;
    for (int i = 1; i < argc; ++i) {
        const char* flag = argv[i];
        if (flag[0] != '-' || flag[1] != '-') {
            snprintf(err, err_len, "unexpected argument '%s'", flag);
            return false;
        }
        if (i + 1 >= argc) {
            snprintf(err, err_len, "missing value for %s", flag);
            return false;
        }
        const char* val = argv[++i];
        long long n = 0;
        if (strcmp(flag, "--queue") == 0) {
            if (strcmp(val, "lockfree") == 0) cfg->queue = QUEUE_LOCKFREE;
            else if (strcmp(val, "mutex") == 0) cfg->queue = QUEUE_MUTEX;
            else { snprintf(err, err_len, "--queue must be lockfree or mutex, got '%s'", val); return false; }
        } else if (strcmp(flag, "--wait") == 0) {
            if (strcmp(val, "spin") == 0) cfg->wait = WAIT_SPIN;
            else if (strcmp(val, "block") == 0) cfg->wait = WAIT_BLOCK;
            else { snprintf(err, err_len, "--wait must be spin or block, got '%s'", val); return false; }
        } else if (strcmp(flag, "--transport") == 0) {
            if (strcmp(val, "sync") == 0) cfg->transport = TRANSPORT_SYNC;
            else if (strcmp(val, "async") == 0) cfg->transport = TRANSPORT_ASYNC;
            else if (strcmp(val, "zerocopy") == 0) cfg->transport = TRANSPORT_ZEROCOPY;
            else { snprintf(err, err_len, "--transport must be sync, async or zerocopy, got '%s'", val); return false; }
        } else if (strcmp(flag, "--chunk-ms") == 0) {
            if (!ab_parse_ll(val, &n) || (n != 10 && n != 20 && n != 40 && n != 80)) {
                snprintf(err, err_len, "--chunk-ms must be 10, 20, 40 or 80, got '%s'", val);
                return false;
            }
            cfg->chunk_ms = (int)n;
        } else if (strcmp(flag, "--work") == 0) {
            if (strcmp(val, "none") == 0) cfg->work = WORK_NONE;
            else if (strcmp(val, "light") == 0) cfg->work = WORK_LIGHT;
            else if (strcmp(val, "heavy") == 0) cfg->work = WORK_HEAVY;
            else { snprintf(err, err_len, "--work must be none, light or heavy, got '%s'", val); return false; }
        } else if (strcmp(flag, "--pace-us") == 0) {
            if (!ab_parse_ll(val, &n) || n < 0 || n > 1000000) {
                snprintf(err, err_len, "--pace-us must be 0..1000000, got '%s'", val);
                return false;
            }
            cfg->pace_us = (int)n;
        } else if (strcmp(flag, "--chunks") == 0) {
            if (!ab_parse_ll(val, &n) || n < 1) {
                snprintf(err, err_len, "--chunks must be >= 1, got '%s'", val);
                return false;
            }
            cfg->chunks = (unsigned long long)n;
        } else if (strcmp(flag, "--warmup") == 0) {
            if (!ab_parse_ll(val, &n) || n < 0) {
                snprintf(err, err_len, "--warmup must be >= 0, got '%s'", val);
                return false;
            }
            cfg->warmup = (unsigned long long)n;
        } else if (strcmp(flag, "--csv") == 0) {
            if (strlen(val) >= sizeof(cfg->csv_path)) {
                snprintf(err, err_len, "--csv path too long");
                return false;
            }
            snprintf(cfg->csv_path, sizeof(cfg->csv_path), "%s", val);
        } else {
            snprintf(err, err_len, "unknown flag '%s'", flag);
            return false;
        }
    }
    // Cross-flag checks run after all flags are parsed, so order does not matter.
    if (cfg->queue == QUEUE_MUTEX && cfg->transport == TRANSPORT_ZEROCOPY) {
        snprintf(err, err_len, "mutex queue with zerocopy transport is not a valid combination");
        return false;
    }
    if (cfg->queue == QUEUE_LOCKFREE && cfg->wait == WAIT_BLOCK) {
        snprintf(err, err_len, "--wait block requires the mutex queue (lockfree queue only spins)");
        return false;
    }
    return true;
}

struct RunResult {
    LatencyStats lat;
    double gpu_work_us;  // mean in-kernel time per measured chunk in process_chunk_warp
    double chunks_per_s;
    double mb_per_s;
    unsigned long long mismatches;
    unsigned long long samples;
    unsigned long long consumed;  // chunks the GPU consumed, warmup included (not in the CSV)
    unsigned long long episodes;
    int stalled;
};

struct PlatformInfo {
    char platform[16];
    char gpu[128];
    char cuda[16];
};

inline PlatformInfo get_platform_info() {
    PlatformInfo p;
    memset(&p, 0, sizeof(p));
#if defined(_WIN32)
    snprintf(p.platform, sizeof(p.platform), "windows");
#else
    snprintf(p.platform, sizeof(p.platform), "linux");
#endif
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        snprintf(p.gpu, sizeof(p.gpu), "%s", prop.name);
    else
        snprintf(p.gpu, sizeof(p.gpu), "unknown");
    int ver = 0;
    cudaRuntimeGetVersion(&ver);
    snprintf(p.cuda, sizeof(p.cuda), "%d.%d", ver / 1000, (ver % 1000) / 10);
    return p;
}

// Grid: V1 mutex+sync, V2 mutex+async, V3 lockfree+sync, V4 lockfree+async, V5 lockfree+zerocopy.
inline const char* variant_name(const RunConfig& c) {
    if (c.transport == TRANSPORT_ZEROCOPY) return "V5";
    if (c.queue == QUEUE_MUTEX) return c.transport == TRANSPORT_SYNC ? "V1" : "V2";
    return c.transport == TRANSPORT_SYNC ? "V3" : "V4";
}

#define CSV_HEADER "variant,queue,wait,transport,queue_capacity,staging_slots,chunk_ms,chunk_bytes,work,pacing_us,platform,gpu,cuda,chunks,warmup,episodes,p50_us,p99_us,p999_us,max_us,mean_us,gpu_work_us,chunks_per_s,mb_per_s,mismatches,stalled"

inline bool append_csv_row(const char* path, const RunConfig& c, const PlatformInfo& p, const RunResult& r) {
    bool need_header = true;
    FILE* probe = fopen(path, "rb");
    if (probe) {
        fseek(probe, 0, SEEK_END);
        need_header = (ftell(probe) <= 0);
        fclose(probe);
    }
    FILE* f = fopen(path, "ab");
    if (!f) return false;
    if (need_header) fprintf(f, "%s\n", CSV_HEADER);
    fprintf(f, "%s,%s,%s,%s,%d,%d,%d,%u,%s,%d,%s,\"%s\",%s,%llu,%llu,%llu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%llu,%d\n",
            variant_name(c), queue_str(c.queue), wait_str(c.wait), transport_str(c.transport),
            AB_QUEUE_CAPACITY, AB_STAGING_SLOTS, c.chunk_ms, samples_for_chunk_ms(c.chunk_ms) * 4u,
            work_str(c.work), c.pace_us, p.platform, p.gpu, p.cuda,
            c.chunks, c.warmup, r.episodes,
            r.lat.p50, r.lat.p99, r.lat.p999, r.lat.max, r.lat.mean, r.gpu_work_us,
            r.chunks_per_s, r.mb_per_s, r.mismatches, r.stalled);
    fclose(f);
    return true;
}
