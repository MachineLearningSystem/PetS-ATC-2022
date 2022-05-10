
#include <iostream>

#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"

void GemmCpu(const float* A, const float* B,
             unsigned int hA, unsigned int wA, unsigned int wB,
             float* C) {
  for (unsigned int i = 0; i < hA; ++i)
    for (unsigned int j = 0; j < wB; ++j) {
      double sum = 0;

      for (unsigned int k = 0; k < wA; ++k) {
        double a = A[i * wA + k];
        double b = B[k * wB + j];
        sum += a * b;
      }

      C[i * wB + j] = (float)sum;
    }
}

int GemmCublas(const float* h_A, const float* h_B,
               int m, int k, int n, bool transA, bool transB,
               int n_iter,
               float* h_C) {
  // allocate host memory for matrices A and B
  unsigned int size_A = m * k;
  unsigned int mem_size_A = sizeof(float) * size_A;
  unsigned int size_B = k * n;
  unsigned int mem_size_B = sizeof(float) * size_B;

  cublasOperation_t transA_ = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB_ = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  // allocate device memory
  float *d_A, *d_B, *d_C;
  unsigned int size_C = m * n;
  unsigned int mem_size_C = sizeof(float) * size_C;

  checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
  checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
  checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

  // create and start timer
  printf("Computing result using CUBLAS...");

  // execute the kernel
  // CUBLAS version 2.0
  {
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasHandle_t handle;
    cudaEvent_t start, stop;

    checkCublasErrors(cublasCreate(&handle));

    // Perform warmup operation with cublas.
    // Pay attention: as cublas store A, B and C in column-major order,
    // we compute (BA)^T=C^T, instead of AB=C.
    checkCublasErrors(cublasSgemm(handle, transB_, transA_,
                                  n, m, k,
                                  &alpha,
                                  d_B, n,
                                  d_A, k,
                                  &beta,
                                  d_C, n));
    
    // Allocate CUDA events that we'll use for timing
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    for (int j = 0; j < n_iter; j++) {
      checkCublasErrors(cublasSgemm(handle, transB_, transA_,
                                    n, m, k,
                                    &alpha,
                                    d_B, n,
                                    d_A, k,
                                    &beta,
                                    d_C, n));
    }

    printf("done.\n");

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / n_iter;
    double flopsPerMatrixMul = 2.0 * (double)m * (double)n * (double)k;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f)
        / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance: %.2f GFlop/s, Time: %.3f msec, Size: %.0f Ops\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul);

    // copy result from device to host
    // checkCudaErrors(cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    // Destroy the handle
    checkCublasErrors(cublasDestroy(handle));
  }

  printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

  checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));

  return EXIT_SUCCESS;
}
