// Measures us per 20 ms chunk (320 samples) of the stand-in GPU work.
// Usage: calibrate_work [iters ...]   (default: the standard sweep)
#include "aetherbridge/kernels.cuh"

#define CAL_CHUNKS 2000
#define CAL_SAMPLES 320

__global__ void cal_kernel(const Chunk* chunks, int n_chunks, int work_iters, float* sink,
                           unsigned int* bad_total) {
    unsigned int bad = 0;
    for (int k = 0; k < n_chunks; ++k)
        bad += process_chunk_warp(&chunks[k], (unsigned long long)k, CAL_SAMPLES, work_iters, sink);
    if (threadIdx.x == 0) *bad_total = bad;
}

static double measure(const Chunk* d_chunks, int work_iters, float* d_sink, unsigned int* d_bad) {
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    cal_kernel<<<1, 32>>>(d_chunks, CAL_CHUNKS, work_iters, d_sink, d_bad);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    cal_kernel<<<1, 32>>>(d_chunks, CAL_CHUNKS, work_iters, d_sink, d_bad);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return (double)ms * 1000.0 / CAL_CHUNKS;
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);

    static Chunk h[CAL_CHUNKS];
    for (int k = 0; k < CAL_CHUNKS; ++k) fill_chunk(&h[k], (unsigned long long)k, CAL_SAMPLES);
    Chunk* d_chunks;
    float* d_sink;
    unsigned int* d_bad;
    CUDA_CHECK(cudaMalloc(&d_chunks, sizeof(h)));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bad, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_chunks, h, sizeof(h), cudaMemcpyHostToDevice));

    int sweep[8] = {0, 100, 200, 500, 1000, 2000, 5000, 10000};
    int custom[64];
    int count = 8;
    int* list = sweep;
    if (argc > 1) {
        count = 0;
        for (int a = 1; a < argc && count < 64; ++a) custom[count++] = atoi(argv[a]);
        list = custom;
    }
    printf("work_iters   us/chunk\n");
    for (int j = 0; j < count; ++j)
        printf("%10d   %8.2f\n", list[j], measure(d_chunks, list[j], d_sink, d_bad));
    return 0;
}
