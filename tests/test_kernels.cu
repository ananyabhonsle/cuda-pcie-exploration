#include "aetherbridge/kernels.cuh"
#include "check.h"

// One warp processes one chunk; every lane writes its returned count.
__global__ void process_kernel(const Chunk* c, int work_iters, float* sink, unsigned int* out32) {
    unsigned int r = process_chunk_warp(c, work_iters, sink);
    out32[threadIdx.x] = r;
}

__global__ void ack_kernel(unsigned long long* slots) {
    if (threadIdx.x == 0) {
        write_ack(slots, 5ULL);
        write_ack(slots, 4096ULL + 7ULL);
    }
}

static Chunk h_chunk;

// Runs the kernel on a copy of h_chunk; returns lane 0's count, sets *agree.
static unsigned int run_process(int work_iters, bool* agree) {
    Chunk* d_chunk;
    float* d_sink;
    unsigned int* d_out;
    CUDA_CHECK(cudaMalloc(&d_chunk, sizeof(Chunk)));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, 32 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_chunk, &h_chunk, sizeof(Chunk), cudaMemcpyHostToDevice));
    process_kernel<<<1, 32>>>(d_chunk, work_iters, d_sink, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned int out[32];
    CUDA_CHECK(cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost));
    *agree = true;
    for (int l = 1; l < 32; ++l) if (out[l] != out[0]) *agree = false;
    CUDA_CHECK(cudaFree(d_chunk));
    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaFree(d_out));
    return out[0];
}

static void test_valid_chunks_have_no_mismatches() {
    const unsigned int ns[3] = {160, 320, 1280};
    const int iters[2] = {0, 1000};
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 2; ++b) {
            fill_chunk(&h_chunk, 9 + a, ns[a]);
            bool agree;
            CHECK(run_process(iters[b], &agree) == 0u);
            CHECK(agree);
        }
}

static void test_corrupted_samples_are_counted() {
    fill_chunk(&h_chunk, 3, 320);
    h_chunk.samples[0] += 1.0f;
    h_chunk.samples[31] += 1.0f;
    h_chunk.samples[319] += 1.0f;
    bool agree;
    CHECK(run_process(0, &agree) == 3u);
    CHECK(agree);
}

static void test_all_lanes_agree() {
    fill_chunk(&h_chunk, 77, 1280);
    for (int i = 0; i < 1280; i += 7) h_chunk.samples[i] += 0.5f;  // 183 corrupted
    bool agree;
    unsigned int r = run_process(10, &agree);
    CHECK(agree);
    CHECK(r == 183u);
}

static void test_work_iters_for() {
    CHECK(work_iters_for(WORK_NONE) == 0);
    CHECK(work_iters_for(WORK_LIGHT) == AB_WORK_ITERS_LIGHT);
    CHECK(work_iters_for(WORK_HEAVY) == AB_WORK_ITERS_HEAVY);
    CHECK(AB_WORK_ITERS_LIGHT < AB_WORK_ITERS_HEAVY);
}

static void test_ack_slot_written() {
    unsigned long long* h_slots;
    unsigned long long* d_slots;
    CUDA_CHECK(cudaHostAlloc(&h_slots, AB_ACK_SLOTS * sizeof(unsigned long long), cudaHostAllocMapped));
    memset(h_slots, 0, AB_ACK_SLOTS * sizeof(unsigned long long));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_slots, h_slots, 0));
    ack_kernel<<<1, 32>>>(d_slots);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(h_slots[5] == 5ULL);
    CHECK(h_slots[7] == 4103ULL);
    CHECK(h_slots[6] == 0ULL);
    CUDA_CHECK(cudaFreeHost(h_slots));
}

int main() {
    test_valid_chunks_have_no_mismatches();
    test_corrupted_samples_are_counted();
    test_all_lanes_agree();
    test_work_iters_for();
    test_ack_slot_written();
    return check_finish();
}
