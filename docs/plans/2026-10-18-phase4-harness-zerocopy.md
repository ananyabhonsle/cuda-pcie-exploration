# Phase 4: Latency Harness + Zero-Copy Variant (V5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared AetherBridge measurement harness and the V5 (lock-free queue, zero-copy) variant, so `bench` reports per-chunk p50/p99/p99.9 latency and throughput for audio-sized chunks.

**Architecture:** Header-only pieces under `src/aetherbridge/` (chunk type and stats, run config and CSV, the stand-in GPU kernel, the ack observer, the zero-copy transport), compiled into one `bench` binary. A persistent one-warp GPU kernel consumes chunks straight from a ring buffer in mapped pinned memory, in episodes of at most ~800 ms; latency is timed on the CPU clock from "chunk sent" to "GPU ack seen".

**Tech Stack:** C-style C++17, CUDA runtime only (`cuda_runtime.h`, `<cuda/atomic>`), `nvcc -arch=native`; MSVC 14.44 + CUDA 12.9 on Windows, CUDA 13.0 on Colab.

**Spec:** `docs/design/2026-10-18-aetherbridge-design.md` (Phase 4 row of Section 8; Sections 3, 5, 6, 7, 9).

## Global Constraints

- Builds with CUDA 12.9 + MSVC 14.44 (Windows) and CUDA 13.0 (Linux); no libraries beyond the CUDA runtime.
- Shared CPU/GPU state: system-scope `cuda::atomic_ref` loads/stores only, never read-modify-write (`hostNativeAtomicSupported = 0`).
- Every launch that must start promptly is followed by `cudaStreamQuery(0)` (Windows queues launches).
- No kernel runs longer than ~1 s: episodes of `episode_ms = 800` by default.
- Stall watchdog everywhere: no progress for `AB_STALL_TIMEOUT_MS = 1000` → give up, report `stalled = 1`.
- Audio: 16 kHz mono `float32`; chunk sizes 10/20/40/80 ms = 160/320/640/1280 samples. Default 20 ms.
- Queue capacity 64 chunks; ack slots A = 4,096; warm-up 500 chunks; default 20,000 measured chunks; default pacing 1,000 us.
- One warp (32 threads) processes each chunk.
- `src/phase1..3` files are not modified. Commit messages and docs contain no AI attribution.

## Review Focus

1. **Non-default chunk sizes (10 ms, 80 ms):** must pass integrity like 20 ms. Pinned by `test_zerocopy` case `chunk_sizes` (Task 5).
2. **Runs spanning many episodes and wrapping the ack ring (> 4,096 chunks):** no lost or duplicated chunks. Pinned by `test_zerocopy` case `many_episodes_and_ack_wrap` (Task 5).
3. **Invalid flag combinations** (`--queue mutex --transport zerocopy`, `--chunk-ms 30`, unknown flags, missing values): clear error, exit 1, no run. Pinned by `test_config` case `parse_rejects_bad_input` (Task 2).
4. **GPU stops consuming, producer stops sending, or the kernel never starts:** `stalled = 1` within ~1.5 s, no hang, no GPU reset. Pinned by `test_zerocopy` cases `stall_is_reported`, `no_producer_is_reported`, `no_kernel_is_reported` (Task 5).
5. **Out-of-order acks** (needed for Phase 5 async): observer records every chunk exactly once. Pinned by `test_harness` case `out_of_order_acks` (Task 4).

## Build and test convention

All commands run from the repo root. On Windows, first open a terminal with:
`Import-Module D:\VS\BuildTools\Common7\Tools\Microsoft.VisualStudio.DevShell.dll; Enter-VsDevShell -VsInstallPath D:\VS\BuildTools -DevCmdArguments '-arch=x64' -SkipAutomaticLocation`

Build a test: `nvcc -O2 -std=c++17 -arch=native -I src tests/<name>.cu -o build/<name>` (create `build/` once; it is gitignored). Run: `build/<name>` (Windows: `build\<name>.exe`). A test binary prints `ALL PASSED` and exits 0, or prints each failure and exits 1.

## File structure

| File | Responsibility |
|---|---|
| `src/aetherbridge/common.h` | constants, `Chunk`, sample generator, `now_us`, `cpu_relax`, percentiles, `CUDA_CHECK` |
| `src/aetherbridge/config.h` | enums, `RunConfig`, `parse_args`, `RunResult`, platform info, CSV writer |
| `src/aetherbridge/kernels.cuh` | `process_chunk_warp`, `write_ack`, calibrated work levels |
| `src/aetherbridge/harness.h` | `LatencyLog`, `record_send`, `ack_observer`, `pace_until`, `extract_latencies` |
| `src/aetherbridge/zerocopy.cuh` | `ZcRing`, persistent consumer kernel, `run_zerocopy` |
| `src/aetherbridge/bench.cu` | `main`: parse, dispatch, print, append CSV |
| `tests/check.h` | `CHECK` macro and pass/fail exit |
| `tests/test_*.cu`, `tests/calibrate_work.cu` | unit and integration tests, calibration tool |

---

### Task 1: Chunk type, sample generator, stats

**Files:**
- Create: `tests/check.h`, `src/aetherbridge/common.h`, `tests/test_common.cu`

**Interfaces:**
- Produces:
  - Constants: `AB_SAMPLE_RATE 16000`, `AB_MAX_CHUNK_SAMPLES 1280`, `AB_QUEUE_CAPACITY 64`, `AB_STAGING_SLOTS 3`, `AB_ACK_SLOTS 4096`, `AB_ACK_EMPTY 0xFFFFFFFFFFFFFFFFULL`, `AB_STALL_TIMEOUT_MS 1000`.
  - `CUDA_CHECK(call)` (same behaviour as Phase 3: print call, file, line, `cudaGetErrorString`, `exit(EXIT_FAILURE)`).
  - `struct Chunk { unsigned long long seq; unsigned int num_samples; unsigned int pad; float samples[AB_MAX_CHUNK_SAMPLES]; };`
  - `__host__ __device__ inline float sample_value(unsigned long long seq, unsigned int i);`
  - `inline void fill_chunk(Chunk* c, unsigned long long seq, unsigned int num_samples);`
  - `inline unsigned int samples_for_chunk_ms(int chunk_ms);` returns `chunk_ms * 16`.
  - `inline double now_us();` (`steady_clock`, microseconds).
  - `inline void cpu_relax();` (`_mm_pause()` via `<immintrin.h>` on x86; empty elsewhere).
  - `struct LatencyStats { double p50, p99, p999, max, mean; };`
  - `inline LatencyStats compute_stats(double* samples, size_t n);` sorts `samples` in place.
  - `tests/check.h`: `CHECK(cond)` increments a global failure count and prints `FAIL file:line: cond`; `int check_finish()` prints `ALL PASSED` or the count and returns the exit code.

- [ ] **Step 1: Write `tests/test_common.cu`** with these cases:

```cpp
// stats_nearest_rank: values 1..100 in shuffled order
LatencyStats s = compute_stats(v, 100);
CHECK(s.p50 == 50.0); CHECK(s.p99 == 99.0); CHECK(s.p999 == 100.0);
CHECK(s.max == 100.0); CHECK(s.mean == 50.5);
// stats_single_value: {7.0} -> all five fields == 7.0
// samples_for_chunk_ms: 10->160, 20->320, 40->640, 80->1280
// sample_value_host_equals_device: a kernel writes sample_value(seq, i) for
//   seq in {0, 1, 4097, 123456789} and i in [0, 1280) to device memory;
//   host compares bit-for-bit (memcmp) with its own sample_value results.
// sample_value_range: every value in [-1.0f, 1.0f)
// fill_chunk: fill_chunk(&c, 42, 320) -> c.seq == 42, c.num_samples == 320,
//   c.samples[i] == sample_value(42, i) for all i < 320
```

- [ ] **Step 2: Build and run; expect a compile failure** (`common.h` missing).

- [ ] **Step 3: Implement `tests/check.h` and `src/aetherbridge/common.h`.** Percentiles use nearest rank: value at sorted index `ceil(p * n) - 1`, clamped to `[0, n-1]`. The generator must be exact on host and device, so it uses only integer ops and a power-of-two division:

```cpp
unsigned int h = (unsigned int)(seq * 2654435761ULL) ^ (i * 40503u);
return (float)(h & 0xFFFFu) / 32768.0f - 1.0f;
```

- [ ] **Step 4: Build and run `test_common`; expect `ALL PASSED`.**

- [ ] **Step 5: Commit** `feat(phase4): add chunk type, sample generator and latency stats`

---

### Task 2: Run config, argument parsing, CSV

**Files:**
- Create: `src/aetherbridge/config.h`, `tests/test_config.cu`

**Interfaces:**
- Consumes: `LatencyStats`, `samples_for_chunk_ms`, `AB_QUEUE_CAPACITY`, `AB_STAGING_SLOTS` (Task 1).
- Produces:
  - `enum QueueKind { QUEUE_LOCKFREE, QUEUE_MUTEX }; enum WaitKind { WAIT_SPIN, WAIT_BLOCK }; enum Transport { TRANSPORT_SYNC, TRANSPORT_ASYNC, TRANSPORT_ZEROCOPY }; enum WorkLevel { WORK_NONE, WORK_LIGHT, WORK_HEAVY };`
  - `struct RunConfig { QueueKind queue; WaitKind wait; Transport transport; int chunk_ms; WorkLevel work; int pace_us; unsigned long long chunks; unsigned long long warmup; int episode_ms; char csv_path[512]; long long debug_corrupt_seq; long long debug_kernel_stop_at; long long debug_producer_stop_at; int debug_skip_kernel; };` (the `debug_*` fields are test hooks; -1 / 0 = off, never set by `parse_args`)
  - `inline RunConfig default_config();` → lockfree, spin, zerocopy, 20, none, 1000, 20000, 500, 800, `""`, -1, -1, -1, 0.
  - `inline bool parse_args(int argc, char** argv, RunConfig* cfg, char* err, size_t err_len);`
  - `struct RunResult { LatencyStats lat; double chunks_per_s; double mb_per_s; unsigned long long mismatches; unsigned long long samples; unsigned long long episodes; int stalled; };`
  - `struct PlatformInfo { char platform[16]; char gpu[128]; char cuda[16]; };` and `inline PlatformInfo get_platform_info();` (`"windows"`/`"linux"`; `cudaGetDeviceProperties` name; `cudaRuntimeGetVersion` as `"12.9"`).
  - `inline const char* variant_name(const RunConfig& c);` → `"V1"`..`"V5"` per the spec grid.
  - `inline const char* queue_str(QueueKind)`, `wait_str`, `transport_str`, `work_str`: the flag spellings.
  - `inline bool append_csv_row(const char* path, const RunConfig& c, const PlatformInfo& p, const RunResult& r);`

- [ ] **Step 1: Write `tests/test_config.cu`** with these cases:

```cpp
// parse_defaults: argv = {"bench"} -> returns true, equals default_config()
// parse_all_flags: {"bench","--queue","mutex","--wait","block","--transport","async",
//   "--chunk-ms","80","--work","heavy","--pace-us","0","--chunks","5000",
//   "--warmup","100","--csv","out.csv"} -> each field set accordingly
// parse_rejects_bad_input: each returns false with non-empty err:
//   {"--queue","mutex","--transport","zerocopy"}   (err mentions "mutex")
//   {"--queue","lockfree","--wait","block"}
//   {"--chunk-ms","30"}, {"--work","medium"}, {"--pace-us","-5"},
//   {"--chunks","0"}, {"--bogus"}, {"--chunks"} (missing value)
// variant_names: lockfree+zerocopy -> "V5", mutex+sync -> "V1", lockfree+async -> "V4"
// csv_header_and_rows: delete temp file, append twice -> file has 3 lines;
//   line 1 equals CSV_HEADER exactly; row 1 starts with "V5,lockfree,spin,zerocopy,64,3,20,1280,none,1000,";
//   the gpu field is wrapped in double quotes
```

`CSV_HEADER` (exact, defined in `config.h`):
`variant,queue,wait,transport,queue_capacity,staging_slots,chunk_ms,chunk_bytes,work,pacing_us,platform,gpu,cuda,chunks,warmup,episodes,p50_us,p99_us,p999_us,max_us,mean_us,chunks_per_s,mb_per_s,mismatches,stalled`

- [ ] **Step 2: Build and run; expect a compile failure.**

- [ ] **Step 3: Implement `src/aetherbridge/config.h`.** `parse_args` accepts combinations later phases will implement (sync, async, mutex); only the spec's impossible or invalid ones are rejected. `--wait` defaults to `spin`. `append_csv_row` writes the header only when the file is new or empty; latencies `%.2f`, throughput `%.1f`; `chunk_bytes = samples_for_chunk_ms(chunk_ms) * 4`.

- [ ] **Step 4: Build and run `test_config`; expect `ALL PASSED`.**

- [ ] **Step 5: Commit** `feat(phase4): add run config, argument parsing and CSV output`

---

### Task 3: Stand-in GPU kernel and work calibration

**Files:**
- Create: `src/aetherbridge/kernels.cuh`, `tests/test_kernels.cu`, `tests/calibrate_work.cu`

**Interfaces:**
- Consumes: `Chunk`, `sample_value`, `AB_ACK_SLOTS` (Task 1); `WorkLevel` (Task 2).
- Produces:
  - `__device__ unsigned int process_chunk_warp(const Chunk* c, int work_iters, float* sink);` called by all 32 lanes of one warp; returns the chunk's total mismatch count on every lane.
  - `__device__ void write_ack(unsigned long long* ack_slots, unsigned long long seq);` lane 0 only: system-scope release store of `seq` into `ack_slots[seq % AB_ACK_SLOTS]`.
  - `#define AB_WORK_ITERS_LIGHT <calibrated>` and `#define AB_WORK_ITERS_HEAVY <calibrated>`; `__host__ __device__ inline int work_iters_for(WorkLevel w);` (none → 0).

- [ ] **Step 1: Write `tests/test_kernels.cu`.** A test kernel (32 threads) runs `process_chunk_warp` on chunks in device memory and writes the result; a second test kernel calls `write_ack`.

```cpp
// valid_chunks_have_no_mismatches: n in {160, 320, 1280}, work_iters in {0, 1000} -> 0
// corrupted_samples_are_counted: n = 320, samples[0], [31], [319] each += 1.0f -> 3
// all_lanes_agree: every lane returns the same count (kernel writes 32 results)
// ack_slot_written: write_ack(slots, 5) and write_ack(slots, 4096 + 7) on zeroed
//   mapped memory -> slots[5] == 5, slots[7] == 4103
```

- [ ] **Step 2: Build and run; expect a compile failure.**

- [ ] **Step 3: Implement `kernels.cuh`.** Lane `l` handles samples `l, l+32, ...`; mismatches combined with `__reduce_add_sync` (or a shuffle reduction) over the full mask. Work: for each sample, `work_iters` dependent `fmaf` steps into a per-lane accumulator; write the accumulator to `*sink` only if it equals an impossible sentinel, so the compiler cannot delete the loop. Start with placeholder iteration counts.

- [ ] **Step 4: Build and run `test_kernels`; expect `ALL PASSED`.**

- [ ] **Step 5: Write `tests/calibrate_work.cu`:** for 20 ms chunks (320 samples), launch a one-warp kernel processing 2,000 chunks back to back in device memory, for `work_iters` in {0, 100, 200, 500, 1000, 2000, 5000, 10000}, and print microseconds per chunk.

- [ ] **Step 6: Run it on the laptop; set `AB_WORK_ITERS_LIGHT` and `AB_WORK_ITERS_HEAVY`** to the counts giving ~10 us and ~200 us per 20 ms chunk. Put the measured values in a comment next to the defines (GPU name, us/chunk). Work scales linearly with chunk length; note that in the same comment. Re-run `test_kernels`: `ALL PASSED`.

- [ ] **Step 7: Commit** `feat(phase4): add stand-in GPU kernel with calibrated work levels`

---

### Task 4: Ack observer and pacing

**Files:**
- Create: `src/aetherbridge/harness.h`, `tests/test_harness.cu`

**Interfaces:**
- Consumes: `now_us`, `cpu_relax`, `AB_ACK_SLOTS`, `AB_ACK_EMPTY`, `AB_STALL_TIMEOUT_MS` (Task 1).
- Produces:
  - `struct LatencyLog { unsigned long long total; double* t_send; double* t_ack; unsigned char* acked; std::atomic<unsigned long long> sent; std::atomic<int> abort; };`
  - `inline bool latency_log_init(LatencyLog* log, unsigned long long total);` / `inline void latency_log_free(LatencyLog* log);`
  - `inline void record_send(LatencyLog* log, unsigned long long seq);` writes `t_send[seq] = now_us()`, then `sent.store(seq + 1, release)`.
  - `inline void ack_observer(LatencyLog* log, unsigned long long* ack_slots, int* stalled_out);` thread body.
  - `inline void pace_until(double deadline_us);` spins with `cpu_relax` until `now_us() >= deadline_us`.
  - `inline size_t extract_latencies(const LatencyLog* log, unsigned long long warmup, double* out);` writes `t_ack - t_send` for `seq` in `[warmup, total)`, returns the count.

- [ ] **Step 1: Write `tests/test_harness.cu`** (CPU only: a plain host array stands in for the mapped ack slots; a "fake GPU" thread writes acks):

```cpp
// in_order_acks: 1000 chunks, fake GPU acks each after its send -> stalled == 0,
//   all acked, every t_ack >= t_send, extract_latencies(log, 100, out) == 900
// out_of_order_acks: 1000 chunks sent in order; fake GPU acks in reversed blocks
//   of 8 (7,6,...,0,15,14,...) -> stalled == 0, every seq acked exactly once
// ack_ring_wrap: 10000 chunks (> 4096) -> stalled == 0, all acked
// stall_is_reported: 10 sent, only 5 acked -> observer returns within 1.5 s
//   with stalled == 1
// pace_until_waits: pace_until(now_us() + 2000) returns after >= 2000 us
```

- [ ] **Step 2: Build and run; expect a compile failure.**

- [ ] **Step 3: Implement `harness.h`.** Ack slots are read with host-side `cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>` acquire loads. The observer loop:
  1. Load `sent` (acquire). Scan `seq` from `lowest_unacked` to `sent - 1`; for each not yet acked whose slot `ack_slots[seq % AB_ACK_SLOTS] == seq`, set `t_ack[seq] = now_us()` and `acked[seq] = 1`.
  2. Advance `lowest_unacked` past acked entries; return when it reaches `total`.
  3. Return early if `abort` is set.
  4. Track the time of the last new ack. Only while chunks are outstanding (`sent > lowest_unacked`), no new ack for `AB_STALL_TIMEOUT_MS` → `*stalled_out = 1`, return. Time with nothing outstanding (episode gaps) never counts as a stall.

- [ ] **Step 4: Build and run `test_harness`; expect `ALL PASSED`.**

- [ ] **Step 5: Commit** `feat(phase4): add ack observer, pacing and latency extraction`

---

### Task 5: Zero-copy transport (V5) with episodes

**Files:**
- Create: `src/aetherbridge/zerocopy.cuh`, `tests/test_zerocopy.cu`

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces:
  - `struct ZcRing { alignas(64) unsigned long long head; alignas(64) unsigned long long tail; alignas(64) unsigned long long stop_at; alignas(64) unsigned long long running_episode; alignas(64) Chunk slots[AB_QUEUE_CAPACITY]; };` `head`/`tail` are monotonic counters (slot = index % 64; full when `head - tail == 64`), unlike Phase 3's wrapped indices.
  - `struct ZcKernelResult { unsigned long long consumed; unsigned long long mismatches; int stalled; };`
  - `__global__ void zc_consumer_kernel(ZcRing* ring, unsigned long long* ack_slots, int work_iters, unsigned long long episode_id, long long debug_stop_at, ZcKernelResult* out, float* sink);` launched `<<<1, 32>>>`.
  - `inline RunResult run_zerocopy(const RunConfig& cfg);`

- [ ] **Step 1: Write `tests/test_zerocopy.cu`** using small runs through `run_zerocopy`:

```cpp
// basic_paced: chunks 2000, warmup 100, pace 200 us, work none
//   -> mismatches 0, stalled 0, samples == 2000, lat.p50 > 0, p50 <= p99 <= p999 <= max
// chunk_sizes: chunk_ms 10 and 80, chunks 1000, unpaced -> mismatches 0, stalled 0
// many_episodes_and_ack_wrap: chunks 10000, unpaced, work light, episode_ms 20
//   -> episodes > 1, samples == 10000, mismatches 0, stalled 0
// corruption_is_detected: debug_corrupt_seq 500, chunks 1000 -> mismatches == 1
// stall_is_reported (no consumer): debug_kernel_stop_at 300, chunks 1000
//   -> stalled == 1, run_zerocopy returns within 3 s of wall time
// no_producer_is_reported: debug_producer_stop_at 300 (producer stops sending
//   without setting stop_at) -> kernel watchdog fires, stalled == 1, within 3 s
// no_kernel_is_reported: debug_skip_kernel 1 -> the wait for running_episode
//   times out, stalled == 1, within 3 s
```

- [ ] **Step 2: Build and run; expect a compile failure.**

- [ ] **Step 3: Implement `zc_consumer_kernel`.** Same warp discipline as Phase 3c: only lane 0 touches `head`/`tail`/`stop_at`; `__syncwarp` orders the other lanes' reads after lane 0's acquire and before its release; `__shfl_sync` broadcasts values and decisions.
  1. On entry, lane 0 release-stores `running_episode = episode_id`.
  2. Loop. If `tail < head`: all lanes run `process_chunk_warp` on `slots[tail % 64]`; `__syncwarp`; lane 0 does `write_ack(chunk.seq)` then release-stores `tail + 1`. Acks are written per chunk, not per batch.
  3. If empty: lane 0 acquire-loads `stop_at`; if `tail >= stop_at`, exit normally. Otherwise apply the stall watchdog (`globaltimer`, `AB_STALL_TIMEOUT_MS`): set `stalled`, exit.
  4. Test hook: when `debug_stop_at >= 0` and the next chunk's seq equals it, stop consuming but keep looping, so the watchdogs fire.
  5. On exit, lane 0 writes `consumed`, `mismatches`, `stalled` to `out`.

- [ ] **Step 4: Implement `run_zerocopy`.**
  1. Allocate `ZcRing` and the `AB_ACK_SLOTS` ack array with `cudaHostAllocMapped`; zero the ring; fill acks with `AB_ACK_EMPTY`. `cudaMalloc` the `ZcKernelResult` and sink. `latency_log_init(total = warmup + chunks)`. Start the `ack_observer` thread.
  2. Per episode: set `stop_at = AB_ACK_EMPTY` (no stop), `cudaMemset` the result, launch the kernel, `cudaGetLastError`, `cudaStreamQuery(0)`. Wait (with watchdog) until `running_episode == episode_id`, so kernel start-up is never inside a latency sample.
  3. Send until `total` is reached or `episode_ms` has passed since the episode started. Paced: `pace_until(episode_start + k * pace_us)` for the k-th chunk of the episode. For each chunk: wait while the ring is full (spin, watchdog), `record_send(seq)`, `fill_chunk` straight into `slots[head % 64]` (test hook: if `seq == debug_corrupt_seq`, add `1.0f` to `samples[0]`), release-store `head + 1`. Test hooks: if `seq == debug_producer_stop_at`, stop sending and wait for the kernel without setting `stop_at`; if `debug_skip_kernel`, skip the launch in step 2.
  4. Release-store `stop_at = head`; `cudaDeviceSynchronize`. Active time += now − time of the episode's first send. Add the kernel result's mismatches; if it stalled, set `log.abort` and stop.
  5. Join the observer. Fill `RunResult`: latencies via `extract_latencies` + `compute_stats`; `chunks_per_s = total / active_seconds`; `mb_per_s = chunks_per_s * chunk_bytes / 1e6`; `episodes`; `stalled` = kernel, producer or observer stall, or `samples != chunks`.
  6. Free everything on every path.

- [ ] **Step 5: Build and run `test_zerocopy`; expect `ALL PASSED`.** Run it 3 times; it must pass every time.

- [ ] **Step 6: Commit** `feat(phase4): add zero-copy transport (V5) with episodes`

---

### Task 6: `bench` binary, verification runs, README

**Files:**
- Create: `src/aetherbridge/bench.cu`
- Modify: `README.md` (status line for Phase 4, Build & Run section)

**Interfaces:**
- Consumes: `parse_args`, `run_zerocopy`, `get_platform_info`, `append_csv_row`, `variant_name` and the `*_str` helpers.

- [ ] **Step 1: Implement `main(int argc, char** argv)` in `bench.cu`:**
  1. `parse_args`; on failure print `[FATAL] <err>` and a usage line, return 1.
  2. If the transport is `sync` or `async`, print `[FATAL] --transport <t> is implemented in Phase 5` and return 1. If the queue is `mutex`: `... in Phase 6`, return 1.
  3. Otherwise call `run_zerocopy`.
  4. Print the summary in this format:

```
--- AETHERBRIDGE BENCH: V5 (lock-free queue, zero-copy) ---
chunk 20 ms (1280 B), work light, paced every 1000 us, 20000 chunks (+500 warmup), 23 episodes
latency us: p50 X  p99 X  p99.9 X  max X  mean X
throughput: X chunks/s, X MB/s
[SUCCESS] 0 mismatches, no stalls
```

  On failure, print `[FATAL]` lines for mismatches or stalls instead. If `--csv` was given, append the row. Return 0 only when `mismatches == 0 && !stalled`.

- [ ] **Step 2: Build:** `nvcc -O2 -std=c++17 -arch=native -I src src/aetherbridge/bench.cu -o build/bench`.

- [ ] **Step 3: Check that the error paths exit 1 with a clear message:**
  - `build/bench --queue mutex --transport zerocopy`
  - `build/bench --transport sync`
  - `build/bench --chunk-ms 30`

- [ ] **Step 4: Run the laptop verification** (on mains power, other applications closed), appending to `build/phase4_laptop.csv`:
  - `--work none`, `--work light`, `--work heavy`, each paced at 1000 us.
  - `--work none --pace-us 0` (unpaced).

  Expected: every run `[SUCCESS]`, 0 mismatches. Paced `work none` p50 in the low microseconds, consistent with Phase 3's ~1 us per PCIe read. `light` and `heavy` p50 higher by roughly their calibrated GPU time.

- [ ] **Step 5: Update `README.md`:**
  - Phase 4 status line, with the laptop p50/p99 for 20 ms chunks at the three work levels.
  - Build & Run commands for `bench` and the tests.
  - "Next" line pointing to Phase 5.

- [ ] **Step 6: Commit** `feat(phase4): add bench binary, results and README`

- [ ] **Step 7: Colab T4 run (done by the contributor; spec requires it before merging).** Clone the branch, build `bench` with the Step 2 command, and run the four Step 4 configurations. Every run must report `[SUCCESS]`. Add the T4 numbers to the README in a follow-up commit before opening the PR.
