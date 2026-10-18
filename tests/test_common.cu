#include "aetherbridge/common.h"
#include "check.h"

// Writes sample_value on the device for each (seq, i) pair.
__global__ void gen_kernel(const unsigned long long* seqs, int nseq, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nseq * AB_MAX_CHUNK_SAMPLES) return;
    int s = idx / AB_MAX_CHUNK_SAMPLES;
    unsigned int i = (unsigned int)(idx % AB_MAX_CHUNK_SAMPLES);
    out[idx] = sample_value(seqs[s], i);
}

static void test_stats_nearest_rank() {
    double v[100];
    // Shuffle 1..100 with a coprime stride so order is not sorted.
    for (int k = 0; k < 100; ++k) v[(k * 37) % 100] = (double)(k + 1);
    LatencyStats s = compute_stats(v, 100);
    CHECK(s.p50 == 50.0); CHECK(s.p99 == 99.0); CHECK(s.p999 == 100.0);
    CHECK(s.max == 100.0); CHECK(s.mean == 50.5);
}

static void test_stats_single_value() {
    double v[1] = {7.0};
    LatencyStats s = compute_stats(v, 1);
    CHECK(s.p50 == 7.0); CHECK(s.p99 == 7.0); CHECK(s.p999 == 7.0);
    CHECK(s.max == 7.0); CHECK(s.mean == 7.0);
}

static void test_samples_for_chunk_ms() {
    CHECK(samples_for_chunk_ms(10) == 160u);
    CHECK(samples_for_chunk_ms(20) == 320u);
    CHECK(samples_for_chunk_ms(40) == 640u);
    CHECK(samples_for_chunk_ms(80) == 1280u);
}

static void test_sample_value_host_equals_device() {
    const unsigned long long seqs[4] = {0ULL, 1ULL, 4097ULL, 123456789ULL};
    const int total = 4 * AB_MAX_CHUNK_SAMPLES;
    unsigned long long* d_seqs;
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_seqs, sizeof(seqs)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_seqs, seqs, sizeof(seqs), cudaMemcpyHostToDevice));
    gen_kernel<<<(total + 255) / 256, 256>>>(d_seqs, 4, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    static float dev[4 * AB_MAX_CHUNK_SAMPLES];
    static float host[4 * AB_MAX_CHUNK_SAMPLES];
    CUDA_CHECK(cudaMemcpy(dev, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));
    for (int s = 0; s < 4; ++s)
        for (unsigned int i = 0; i < AB_MAX_CHUNK_SAMPLES; ++i)
            host[s * AB_MAX_CHUNK_SAMPLES + i] = sample_value(seqs[s], i);
    CHECK(memcmp(dev, host, sizeof(host)) == 0);
    CUDA_CHECK(cudaFree(d_seqs));
    CUDA_CHECK(cudaFree(d_out));
}

static void test_sample_value_range() {
    bool ok = true;
    for (unsigned long long seq = 0; seq < 50; ++seq)
        for (unsigned int i = 0; i < AB_MAX_CHUNK_SAMPLES; ++i) {
            float f = sample_value(seq, i);
            if (!(f >= -1.0f && f < 1.0f)) ok = false;
        }
    CHECK(ok);
}

static void test_fill_chunk() {
    static Chunk c;
    fill_chunk(&c, 42, 320);
    CHECK(c.seq == 42);
    CHECK(c.num_samples == 320);
    bool ok = true;
    for (unsigned int i = 0; i < 320; ++i)
        if (c.samples[i] != sample_value(42, i)) ok = false;
    CHECK(ok);
}

int main() {
    test_stats_nearest_rank();
    test_stats_single_value();
    test_samples_for_chunk_ms();
    test_sample_value_host_equals_device();
    test_sample_value_range();
    test_fill_chunk();
    return check_finish();
}
