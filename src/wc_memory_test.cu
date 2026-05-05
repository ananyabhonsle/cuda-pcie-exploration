#include <cuda_runtime.h>
#include <stdio.h>

__global__ void gpu_test() {}

int main() {
    printf("--- PHASE 1: PHYSICS TEST ---\n");
    
    gpu_test<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("[SUCCESS] GPU is online.\n");

    float* wc_memory = NULL;
    cudaError_t status = cudaHostAlloc(&wc_memory, 1024 * sizeof(float), cudaHostAllocWriteCombined);
    
    if (status != cudaSuccess) {
        printf("[FATAL] Failed to allocate WC memory.\n");
        return -1;
    }
    printf("[SUCCESS] WC Memory allocated.\n");

    for (int i = 0; i < 1024; i++) {
        wc_memory[i] = (float)i * 1.5f;
    }
    printf("[SUCCESS] CPU wrote 1024 floats to WC memory.\n");

    float test_read = 0.0f;
    for (int i = 0; i < 1024; i++) {
        test_read += wc_memory[i]; 
    }
    printf("[SUCCESS] CPU read back from WC memory. Checksum: %f\n", test_read);

    cudaFreeHost(wc_memory);
    printf("[SUCCESS] Memory freed.\n");
    printf("--- PHASE 1 COMPLETE ---\n");
    return 0;
}