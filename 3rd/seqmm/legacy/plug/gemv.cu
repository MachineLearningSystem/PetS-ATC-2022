#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <iostream>

int bench(int m, int n) {
    int iter = 20;

    float *A, *x, *y;
    cudaMalloc(&A, m * n * sizeof(float));
    cudaMalloc(&x, n * sizeof(float));
    cudaMalloc(&y, m * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.f;
    float beta = 0.f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, A, m, x, 1, &beta, y, 1);

    cudaProfilerStart();
    cudaEventRecord(start);
    for (int i = 0; i < iter; ++i) {
        cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, A, m, x, 1, &beta, y, 1);
    }
    cudaEventRecord(stop);
    cudaProfilerStop();

    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << iter <<  " round cusparseSpMM duration:" << ms << "ms" << std::endl;
    std::cout << "each round on average :" << ms / iter << "ms" << std::endl;

    cudaFree(A);
    cudaFree(x);
    cudaFree(y);
    return 0;
}

int main() {
    int test_case[] = {8192, 1024, 8192, 8192, 24576, 8192, 32768, 8192, 8192, 32768};

    for (int i = 0; i < 5; ++i) {
        bench(test_case[i * 2], test_case[i * 2 + 1]);
    }

    return 0;
}



