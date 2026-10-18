#include "aetherbridge/zerocopy.cuh"
#include "check.h"

static void print_result(const char* name, const RunResult& r, double wall_ms) {
    printf("%-26s p50 %.2f p99 %.2f p99.9 %.2f max %.2f mean %.2f us | %.0f chunks/s %.1f MB/s | "
           "samples %llu episodes %llu mismatches %llu stalled %d | %.0f ms\n",
           name, r.lat.p50, r.lat.p99, r.lat.p999, r.lat.max, r.lat.mean, r.chunks_per_s, r.mb_per_s,
           r.samples, r.episodes, r.mismatches, r.stalled, wall_ms);
}

// Runs one configuration and returns its wall time in ms through *wall_ms.
static RunResult timed_run(const char* name, const RunConfig& cfg, double* wall_ms) {
    double t0 = now_us();
    RunResult r = run_zerocopy(cfg);
    *wall_ms = (now_us() - t0) / 1000.0;
    print_result(name, r, *wall_ms);
    return r;
}

static void test_basic_paced() {
    RunConfig c = default_config();
    c.chunks = 2000;
    c.warmup = 100;
    c.pace_us = 200;
    c.work = WORK_NONE;
    double ms;
    RunResult r = timed_run("basic_paced", c, &ms);
    CHECK(r.mismatches == 0);
    CHECK(r.stalled == 0);
    CHECK(r.samples == 2000);
    CHECK(r.episodes >= 1);
    CHECK(r.lat.p50 > 0.0);
    CHECK(r.lat.p50 <= r.lat.p99);
    CHECK(r.lat.p99 <= r.lat.p999);
    CHECK(r.lat.p999 <= r.lat.max);
    CHECK(r.chunks_per_s > 0.0);
    CHECK(r.mb_per_s > 0.0);
}

static void test_chunk_sizes() {
    const int sizes[2] = {10, 80};
    for (int i = 0; i < 2; ++i) {
        RunConfig c = default_config();
        c.chunk_ms = sizes[i];
        c.chunks = 1000;
        c.pace_us = 0;
        char name[32];
        snprintf(name, sizeof(name), "chunk_sizes_%dms", sizes[i]);
        double ms;
        RunResult r = timed_run(name, c, &ms);
        CHECK(r.mismatches == 0);
        CHECK(r.stalled == 0);
        CHECK(r.samples == 1000);
    }
}

static void test_many_episodes_and_ack_wrap() {
    RunConfig c = default_config();
    c.chunks = 10000;
    c.pace_us = 0;
    c.work = WORK_LIGHT;
    c.episode_ms = 20;
    double ms;
    RunResult r = timed_run("many_episodes_and_ack_wrap", c, &ms);
    CHECK(r.episodes > 1);
    CHECK(r.samples == 10000);
    CHECK(r.mismatches == 0);
    CHECK(r.stalled == 0);
}

static void test_corruption_is_detected() {
    RunConfig c = default_config();
    c.chunks = 1000;
    c.debug_corrupt_seq = 500;
    double ms;
    RunResult r = timed_run("corruption_is_detected", c, &ms);
    CHECK(r.mismatches == 1);
    CHECK(r.stalled == 0);
}

static void test_stall_is_reported() {
    RunConfig c = default_config();
    c.chunks = 1000;
    c.debug_kernel_stop_at = 300;
    double ms;
    RunResult r = timed_run("stall_is_reported", c, &ms);
    CHECK(r.stalled == 1);
    CHECK(ms < 3000.0);
}

static void test_no_producer_is_reported() {
    RunConfig c = default_config();
    c.chunks = 1000;
    c.debug_producer_stop_at = 300;
    double ms;
    RunResult r = timed_run("no_producer_is_reported", c, &ms);
    CHECK(r.stalled == 1);
    CHECK(ms < 3000.0);
}

static void test_no_kernel_is_reported() {
    RunConfig c = default_config();
    c.chunks = 1000;
    c.debug_skip_kernel = 1;
    double ms;
    RunResult r = timed_run("no_kernel_is_reported", c, &ms);
    CHECK(r.stalled == 1);
    CHECK(ms < 3000.0);
}

int main() {
    test_basic_paced();
    test_chunk_sizes();
    test_many_episodes_and_ack_wrap();
    test_corruption_is_detected();
    test_stall_is_reported();
    test_no_producer_is_reported();
    test_no_kernel_is_reported();
    // A healthy run after the stall tests proves the GPU and runtime were left usable.
    test_basic_paced();
    return check_finish();
}
