#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>

#include <cuda_runtime.h>

#include "paisparse_op.h"
#include "cublas_op.h"
#include "utils.h"

template <typename T>
void Bench(const int m, const int n, const int k, const int sparsity) {
  
  printf("[Generate Random Dense Matrix A] - Starting...");
  int A_size = m * k;
  T* A = new T[A_size] ();
  seqmm::GenDenseMatrix(A, m, k);
  printf("done.\n");

  T* d_A;
  CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_A, A, m * k * sizeof(T), cudaMemcpyHostToDevice));

  printf("[Generate Random Sparse Matrix B] - Starting...");
  int B_size = k * n;
  T* B = new T[B_size] ();
  seqmm::GenSparseMatrix(B, k, n, sparsity);
  printf("done.\n");

  T* d_B;
  CHECK_CUDA(cudaMalloc(&d_B, n * k * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_B, B, n * k * sizeof(T), cudaMemcpyHostToDevice));

  T* d_C;
  int C_size = m * n;
  CHECK_CUDA(cudaMalloc(&d_C, C_size * sizeof(T)));

  printf("[Matrix Multiplication with PaiSparse] - Starting...");
  seqmm::PaiSparseMM<T> pai_spmm = seqmm::PaiSparseMM<T>();
  pai_spmm.Init(d_A, d_B, d_C, m, n, k, false, false, 1.0, 0.0);
  pai_spmm.Run();
  printf("done.\n");
  
  T* C = new T[C_size] ();
  CHECK_CUDA(cudaMemcpy(C, d_C, C_size * sizeof(T), cudaMemcpyDeviceToHost));

  printf("[Matrix Multiplication with CuBLAS] - Starting...");
  seqmm::CuBlasMM<T> cublas_gemm = seqmm::CuBlasMM<T>();
  cublas_gemm.Init(1.0f, 0.0f, false, true, m, n, k);
  cublas_gemm.Run(d_A, d_B, d_C, m, n, k);
  printf("done.\n");

  T* C_cublas = new T[C_size] ();
  CHECK_CUDA(cudaMemcpy(C_cublas, d_C, C_size * sizeof(T), cudaMemcpyDeviceToHost));

  bool res = seqmm::CompareL2fe(C_cublas, C, C_size, (T)1.0e-3f);
  printf("Comparing CUSPARSE Matrix Multiply with CPU results: %s\n",
         (true == res) ? "PASS" : "FAIL");
    
  // Clean up.
  pai_spmm.Clear();
  cublas_gemm.Clear();
  
  delete [] A;
  delete [] B;
  delete [] C;
  delete [] C_cublas;

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));  
}

int main(int argc, char **argv) {
  if (argc != 5) {
        printf("Usage: ./spmm_example m k n sparsity\n");
        exit(-1);
  }
  int m = atoi(argv[1]);
  int k = atoi(argv[2]);
  int n = atoi(argv[3]);
  int sparsity = atoi(argv[4]);

  // Bench<float>(m, n, k, sparsity);
  Bench<half>(m, n, k, sparsity);
  
  return 0;
}
