#include "aetherbridge/config.h"
#include "check.h"

static bool parse(std::initializer_list<const char*> args, RunConfig* cfg, char* err, size_t n) {
    static char* argv[32];
    int argc = 1;
    static char prog[] = "bench";
    argv[0] = prog;
    for (const char* a : args) argv[argc++] = const_cast<char*>(a);
    *cfg = default_config();
    err[0] = 0;
    return parse_args(argc, argv, cfg, err, n);
}

static void test_parse_defaults() {
    RunConfig c; char err[256];
    CHECK(parse({}, &c, err, sizeof(err)));
    CHECK(c.queue == QUEUE_LOCKFREE); CHECK(c.wait == WAIT_SPIN);
    CHECK(c.transport == TRANSPORT_ZEROCOPY); CHECK(c.chunk_ms == 20);
    CHECK(c.work == WORK_NONE); CHECK(c.pace_us == 1000);
    CHECK(c.chunks == 20000ULL); CHECK(c.warmup == 500ULL);
    CHECK(c.episode_ms == 800); CHECK(c.csv_path[0] == 0);
    CHECK(c.debug_corrupt_seq == -1); CHECK(c.debug_kernel_stop_at == -1);
    CHECK(c.debug_producer_stop_at == -1); CHECK(c.debug_skip_kernel == 0);
    CHECK(c.debug_kernel_max_age_ms == 0);
    CHECK(c.debug_bad_seq_at == -1); CHECK(c.debug_bad_count_at == -1);
    CHECK(c.debug_zero_slot_at == -1);
}

static void test_parse_all_flags() {
    RunConfig c; char err[256];
    CHECK(parse({"--queue", "mutex", "--wait", "block", "--transport", "async",
                 "--chunk-ms", "80", "--work", "heavy", "--pace-us", "0",
                 "--chunks", "5000", "--warmup", "100", "--csv", "out.csv"},
                &c, err, sizeof(err)));
    CHECK(c.queue == QUEUE_MUTEX); CHECK(c.wait == WAIT_BLOCK);
    CHECK(c.transport == TRANSPORT_ASYNC); CHECK(c.chunk_ms == 80);
    CHECK(c.work == WORK_HEAVY); CHECK(c.pace_us == 0);
    CHECK(c.chunks == 5000ULL); CHECK(c.warmup == 100ULL);
    CHECK(strcmp(c.csv_path, "out.csv") == 0);
}

static void test_parse_rejects_bad_input() {
    RunConfig c; char err[256];
    CHECK(!parse({"--queue", "mutex", "--transport", "zerocopy"}, &c, err, sizeof(err)));
    CHECK(err[0] != 0); CHECK(strstr(err, "mutex") != nullptr);
    CHECK(!parse({"--queue", "lockfree", "--wait", "block"}, &c, err, sizeof(err)));
    CHECK(err[0] != 0);
    CHECK(!parse({"--chunk-ms", "30"}, &c, err, sizeof(err))); CHECK(err[0] != 0);
    CHECK(!parse({"--work", "medium"}, &c, err, sizeof(err))); CHECK(err[0] != 0);
    CHECK(!parse({"--pace-us", "-5"}, &c, err, sizeof(err))); CHECK(err[0] != 0);
    CHECK(!parse({"--chunks", "0"}, &c, err, sizeof(err))); CHECK(err[0] != 0);
    CHECK(!parse({"--bogus"}, &c, err, sizeof(err))); CHECK(err[0] != 0);
    CHECK(!parse({"--chunks"}, &c, err, sizeof(err))); CHECK(err[0] != 0);
}

static void test_variant_names() {
    RunConfig c = default_config();
    CHECK(strcmp(variant_name(c), "V5") == 0);
    c.queue = QUEUE_MUTEX; c.transport = TRANSPORT_SYNC;
    CHECK(strcmp(variant_name(c), "V1") == 0);
    c.transport = TRANSPORT_ASYNC;
    CHECK(strcmp(variant_name(c), "V2") == 0);
    c.queue = QUEUE_LOCKFREE;
    CHECK(strcmp(variant_name(c), "V4") == 0);
    c.transport = TRANSPORT_SYNC;
    CHECK(strcmp(variant_name(c), "V3") == 0);
}

static void test_csv_header_and_rows() {
    const char* path = "build/test_config.csv";
    remove(path);
    RunConfig c = default_config();
    PlatformInfo p = get_platform_info();
    RunResult r;
    memset(&r, 0, sizeof(r));
    r.lat.p50 = 12.345; r.chunks_per_s = 999.95;
    CHECK(append_csv_row(path, c, p, r));
    CHECK(append_csv_row(path, c, p, r));
    FILE* f = fopen(path, "r");
    CHECK(f != nullptr);
    char lines[4][2048]; int n = 0;
    if (f) {
        while (n < 4 && fgets(lines[n], sizeof(lines[n]), f)) {
            size_t l = strlen(lines[n]);
            while (l && (lines[n][l - 1] == '\n' || lines[n][l - 1] == '\r')) lines[n][--l] = 0;
            ++n;
        }
        fclose(f);
    }
    CHECK(n == 3);
    if (n == 3) {
        CHECK(strcmp(lines[0], CSV_HEADER) == 0);
        const char* want = "V5,lockfree,spin,zerocopy,64,3,20,1280,none,1000,";
        CHECK(strncmp(lines[1], want, strlen(want)) == 0);
        char q[160];
        snprintf(q, sizeof(q), ",\"%s\",", p.gpu);
        CHECK(strstr(lines[1], q) != nullptr);
    }
    remove(path);
}

int main() {
    test_parse_defaults();
    test_parse_all_flags();
    test_parse_rejects_bad_input();
    test_variant_names();
    test_csv_header_and_rows();
    return check_finish();
}
