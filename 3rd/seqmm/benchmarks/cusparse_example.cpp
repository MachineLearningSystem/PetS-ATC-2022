#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>

#include <cuda_runtime.h>

#include "cusparse_op.h"
#include "cublas_op.h"
#include "utils.h"

template <typename T>
void Bench(const int m, const int n, const int k, const int sparsity) {
  int A_size = m * k;
  int B_size = k * n;
  int C_size = m * n;
  int B_nnz = 0;

  T* A = new T[A_size]();
  T* B = new T[B_size]();
  T* B_csr_values;
  int *B_csr_col_indices;
  int *B_csr_row_offsets;
  T* C = new T[C_size]();
  T* C_cublas = new T[C_size]();

  T* d_A;
  T* d_B_csr_values;
  int *d_B_csr_col_indices;
  int *d_B_csr_row_offsets;
  T* d_B;
  T* d_C;
  
  cudaStream_t stream;
  cusparseHandle_t handle;
  
  seqmm::CuSparseMM<T> cusparse_spmm = seqmm::CuSparseMM<T>();
  seqmm::CuBlasMM<T> cublas_gemm = seqmm::CuBlasMM<T>();

  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUSPARSE(cusparseCreate(&handle));
  CHECK_CUSPARSE(cusparseSetStream(handle, stream));
  
  CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_B, n * k * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_C, C_size * sizeof(T)));
    
  printf("[Generate Random Dense Matrix A] - Starting...");
  seqmm::GenDenseMatrix(A, m, k);
  printf("done.\n");

  CHECK_CUDA(cudaMemcpy(d_A, A, m * k * sizeof(T), cudaMemcpyHostToDevice));

  printf("[Generate Random Sparse Matrix B] - Starting...");
  seqmm::GenSparseMatrix(B, k, n, sparsity);
  printf("done.\n");

  printf("[Convert Dense Matrix B To Sparse] - Starting...");
  seqmm::SparseMatrixGetSize(B, B_size, &B_nnz);
  B_csr_values = new T[B_nnz] ();
  B_csr_col_indices = new int[B_nnz] ();
  // B_csr_row_offsets = new int[n + 1] ();
  B_csr_row_offsets = new int[k + 1] ();
  // seqmm::Dense2Sparse(B, n, k, 
  //                    B_csr_values, B_csr_col_indices, B_csr_row_offsets);
  seqmm::Dense2Sparse(B, k, n, 
                      B_csr_values, B_csr_col_indices, B_csr_row_offsets);
  printf("done.\n");

  CHECK_CUDA(cudaMemcpy(d_B, B, n * k * sizeof(T), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&d_B_csr_values, B_nnz * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_values, B_csr_values, B_nnz * sizeof(T),
                        cudaMemcpyHostToDevice));
  
  CHECK_CUDA(cudaMalloc(&d_B_csr_col_indices, B_nnz * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_col_indices, B_csr_col_indices,
                        B_nnz * sizeof(int), cudaMemcpyHostToDevice));
  
  CHECK_CUDA(cudaMalloc(&d_B_csr_row_offsets, (n + 1) * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_row_offsets, B_csr_row_offsets,
                        (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  
  printf("[Matrix Multiplication with CuSparse] - Starting...");
  // cusparse_spmm.Run(d_A, d_B_csr_values, d_B_csr_col_indices, d_B_csr_row_offsets,
  //                  B_nnz, m, n, k, false, false, 1.0, 0.0, d_C, handle);
  cusparse_spmm.Run(d_A, d_B_csr_values, d_B_csr_col_indices, d_B_csr_row_offsets,
                    B_nnz, m, n, k, false, true, 1.0, 0.0, d_C, handle);
  printf("done.\n");

  CHECK_CUDA(cudaMemcpy(C, d_C, C_size * sizeof(T), cudaMemcpyDeviceToHost));

  // cublas_gemm.Init(1.0f, 0.0f, false, true, m, n, k);
  cublas_gemm.Init(1.0f, 0.0f, false, false, m, n, k);
  printf("[Matrix Multiplication with CuBLAS] - Starting...");
  cublas_gemm.Run(d_A, d_B, d_C, m, n, k);
  printf("done.\n");
  
  CHECK_CUDA(cudaMemcpy(C_cublas, d_C, C_size * sizeof(T), cudaMemcpyDeviceToHost));
  
  bool res = seqmm::CompareL2fe(C_cublas, C, C_size, (T)1.0e-6f);
  printf("Comparing CUSPARSE Matrix Multiply with CPU results: %s\n",
         (true == res) ? "PASS" : "FAIL");
    
  // Clean up.
  cusparse_spmm.Clear();
  cublas_gemm.Clear();
  
  delete [] A;
  delete [] B;
  delete [] B_csr_col_indices;
  delete [] B_csr_values;
  delete [] B_csr_row_offsets;
  delete [] C;
  delete [] C_cublas;

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_B_csr_values));
  CHECK_CUDA(cudaFree(d_B_csr_col_indices));
  CHECK_CUDA(cudaFree(d_B_csr_row_offsets));
  CHECK_CUDA(cudaFree(d_C));

  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUSPARSE(cusparseDestroy(handle));
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

  Bench<half>(m, n, k, sparsity);
  // Bench<float>(m, n, k, sparsity);
  
  return 0;
}
