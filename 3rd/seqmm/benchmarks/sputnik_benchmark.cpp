#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <numeric>

#include <cuda_runtime.h>

#include "sputnik_op.h"
#include "cublas_op.h"
#include "utils.h"

void BenchSpAxDnB(const int m, const int n, const int k, const int sparsity) {
  int A_size = m * k;
  int B_size = k * n;
  int C_size = m * n;
  int B_nnz = 0;

  float* A = new float[A_size]();
  float* B = new float[B_size]();
  float* B_csr_values;
  int *B_csr_col_indices;
  int *B_csr_row_offsets;
  int *B_csr_row_indices;
  float* C = new float[C_size]();
  float* C_cublas = new float[C_size]();

  float* d_A;
  float* d_B_csr_values;
  int *d_B_csr_col_indices;
  int *d_B_csr_row_offsets;
  int *d_B_csr_row_indices;
  float* d_B;
  float* d_C;
  
  cudaStream_t stream;
  
  seqmm::SputnikMM sputnik_spmm = seqmm::SputnikMM();
  seqmm::CuBlasMM<float> cublas_gemm = seqmm::CuBlasMM<float>();

  CHECK_CUDA(cudaStreamCreate(&stream));
  
  CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_B, n * k * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_C, C_size * sizeof(float)));
    
  printf("[Generate Random Dense Matrix A] - Starting...");
  seqmm::GenDenseMatrix(A, m, k);
  printf("done.\n");

  CHECK_CUDA(cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice));

  printf("[Generate Random Sparse Matrix B] - Starting...");
  seqmm::GenSparseMatrix(B, k, n, sparsity);
  printf("done.\n");

  printf("[Convert Dense Matrix B To Sparse] - Starting...");
  seqmm::SparseMatrixGetSize(B, B_size, &B_nnz);
  B_csr_values = new float[B_nnz] ();
  B_csr_col_indices = new int[B_nnz] ();
  B_csr_row_offsets = new int[n + 1] ();
  B_csr_row_indices = new int[n] ();
  seqmm::Dense2Sparse(B, n, k, 
                      B_csr_values, B_csr_col_indices, B_csr_row_offsets);
  std::iota(B_csr_row_indices, B_csr_row_indices + n, 0);
  printf("done.\n");

  CHECK_CUDA(cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&d_B_csr_values, B_nnz * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_values, B_csr_values, B_nnz * sizeof(float),
                        cudaMemcpyHostToDevice));
  
  CHECK_CUDA(cudaMalloc(&d_B_csr_col_indices, B_nnz * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_col_indices, B_csr_col_indices,
                        B_nnz * sizeof(int), cudaMemcpyHostToDevice));
  
  CHECK_CUDA(cudaMalloc(&d_B_csr_row_offsets, (n + 1) * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_row_offsets, B_csr_row_offsets,
                        (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&d_B_csr_row_indices, n * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_row_indices, B_csr_row_indices,
                        n * sizeof(int), cudaMemcpyHostToDevice));

  int iter = 100;
  printf("[Matrix Multiplication with Sputnik] - Starting...");
  sputnik_spmm.Run(d_A, d_B_csr_values, d_B_csr_col_indices, d_B_csr_row_offsets,
                   d_B_csr_row_indices, B_nnz, m, n, k, d_C, false, stream);
  CUDA_EVENT_START(sputnik, "Sputnik");
  for (int i = 0; i < iter; ++i) {
    sputnik_spmm.Run(d_A, d_B_csr_values, d_B_csr_col_indices, d_B_csr_row_offsets,
                     d_B_csr_row_indices, B_nnz, m, n, k, d_C, false, stream);
  }
  CUDA_EVENT_STOP(sputnik, "Sputnik");
  printf("done.\n");

  CHECK_CUDA(cudaMemcpy(C, d_C, C_size * sizeof(float), cudaMemcpyDeviceToHost));

  cublas_gemm.Init(1.0f, 0.0f, false, false, n, m, k);
  printf("[Matrix Multiplication with CuBLAS] - Starting...");
  cublas_gemm.Run(d_B, d_A, d_C, n, m, k);

  CUDA_EVENT_START(cublas, "CUBLAS");
  for (int i = 0; i < iter; ++i) {
    cublas_gemm.Run(d_B, d_A, d_C, n, m, k);
  }
  CUDA_EVENT_STOP(cublas, "CUBLAS");
  printf("done.\n");
  
  CHECK_CUDA(cudaMemcpy(C_cublas, d_C, C_size * sizeof(float), cudaMemcpyDeviceToHost));
  
  bool res = seqmm::CompareL2fe(C_cublas, C, C_size, (float)1.0e-6f);
  printf("Comparing CUSPARSE Matrix Multiply with CPU results: %s\n",
         (true == res) ? "PASS" : "FAIL");
    
  // Clean up.
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
}

void BenchDnAxSpB(const int m, const int n, const int k, const int sparsity) {
  int A_size = m * k;
  int B_size = n * k;
  int C_size = m * n;
  int B_nnz = 0;

  float* A = new float[A_size]();
  float* A_trans = new float[A_size]();
  float* B = new float[B_size]();
  float* B_trans = new float[B_size]();
  float* B_csr_values;
  int *B_csr_col_indices;
  int *B_csr_row_offsets;
  int *B_csr_row_indices;
  float* C = new float[C_size]();
  float* C_trans = new float[C_size]();
  float* C_cublas = new float[C_size]();

  float* d_A;
  float* d_A_trans;
  float* d_B_csr_values;
  int *d_B_csr_col_indices;
  int *d_B_csr_row_offsets;
  int *d_B_csr_row_indices;
  float* d_B;
  float* d_C;
  
  cudaStream_t stream;
  
  seqmm::SputnikMM sputnik_spmm = seqmm::SputnikMM();
  seqmm::CuBlasMM<float> cublas_gemm = seqmm::CuBlasMM<float>();

  CHECK_CUDA(cudaStreamCreate(&stream));
  
  CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_A_trans, k * m * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_C, C_size * sizeof(float)));
    
  printf("[Generate Random Dense Matrix A] - Starting...");
  seqmm::GenDenseMatrix(A, m, k);
  printf("done.\n");

  printf("[Transpose A to A_trans] - Starting...");
  seqmm::Transpose(A, m, k, A_trans);
  printf("done.\n");

  CHECK_CUDA(cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_A_trans, A_trans, k * m * sizeof(float), cudaMemcpyHostToDevice));

  printf("[Generate Random Sparse Matrix B] - Starting...");
  seqmm::GenSparseMatrix(B, k, n, sparsity);
  printf("done.\n");

  printf("[Transpose B to B_trans] - Starting...");
  seqmm::Transpose(B, k, n, B_trans);
  printf("done.\n");

  printf("[Convert Dense Matrix B To Sparse] - Starting...");
  seqmm::SparseMatrixGetSize(B_trans, B_size, &B_nnz);
  B_csr_values = new float[B_nnz] ();
  B_csr_col_indices = new int[B_nnz] ();
  B_csr_row_offsets = new int[n + 1] ();
  B_csr_row_indices = new int[n] ();
  seqmm::Dense2Sparse(B_trans, n, k, 
                      B_csr_values, B_csr_col_indices, B_csr_row_offsets);
  std::iota(B_csr_row_indices, B_csr_row_indices + n, 0);
  printf("done.\n");

  CHECK_CUDA(cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&d_B_csr_values, B_nnz * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_values, B_csr_values, B_nnz * sizeof(float),
                        cudaMemcpyHostToDevice));
  
  CHECK_CUDA(cudaMalloc(&d_B_csr_col_indices, B_nnz * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_col_indices, B_csr_col_indices,
                        B_nnz * sizeof(int), cudaMemcpyHostToDevice));
  
  CHECK_CUDA(cudaMalloc(&d_B_csr_row_offsets, (n + 1) * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_row_offsets, B_csr_row_offsets,
                        (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&d_B_csr_row_indices, n * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_row_indices, B_csr_row_indices,
                        n * sizeof(int), cudaMemcpyHostToDevice));

  sputnik_spmm.Init();
  
  int iter = 100;
  printf("[Matrix Multiplication with Sputnik] - Starting...");
  sputnik_spmm.Run(d_A_trans, d_B_csr_values, d_B_csr_col_indices, d_B_csr_row_offsets,
                   d_B_csr_row_indices, B_nnz, m, n, k, d_C, false, stream);
  CUDA_EVENT_START(sputnik, "Sputnik");
  for (int i = 0; i < iter; ++i) {
    sputnik_spmm.Run(d_A_trans, d_B_csr_values, d_B_csr_col_indices, d_B_csr_row_offsets,
                     d_B_csr_row_indices, B_nnz, m, n, k, d_C, false, stream);
  }
  CUDA_EVENT_STOP(sputnik, "Sputnik");
  printf("done.\n");

  CHECK_CUDA(cudaMemcpy(C_trans, d_C, C_size * sizeof(float), cudaMemcpyDeviceToHost));

  printf("[Transpose C_trans to C] - Starting...");
  seqmm::Transpose(C_trans, n, m, C);
  printf("done.\n");

  cublas_gemm.Init(1.0f, 0.0f, false, false, m, n, k);
  printf("[Matrix Multiplication with CuBLAS] - Starting...");
  cublas_gemm.Run(d_A, d_B, d_C, m, n, k);

  CUDA_EVENT_START(cublas, "CUBLAS");
  for (int i = 0; i < iter; ++i) {
    cublas_gemm.Run(d_A, d_B, d_C, m, n, k);
  }
  CUDA_EVENT_STOP(cublas, "CUBLAS");
  printf("done.\n");
  
  CHECK_CUDA(cudaMemcpy(C_cublas, d_C, C_size * sizeof(float), cudaMemcpyDeviceToHost));
  
  bool res = seqmm::CompareL2fe(C_cublas, C, C_size, (float)1.0e-6f);
  printf("Comparing CUSPARSE Matrix Multiply with CPU results: %s\n",
         (true == res) ? "PASS" : "FAIL");
    
  // Clean up.
  cublas_gemm.Clear();
  sputnik_spmm.Clear();
  
  delete [] A;
  delete [] A_trans;
  delete [] B;
  delete [] B_trans;
  delete [] B_csr_col_indices;
  delete [] B_csr_values;
  delete [] B_csr_row_offsets;
  delete [] C;
  delete [] C_trans;
  delete [] C_cublas;

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_A_trans));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_B_csr_values));
  CHECK_CUDA(cudaFree(d_B_csr_col_indices));
  CHECK_CUDA(cudaFree(d_B_csr_row_offsets));
  CHECK_CUDA(cudaFree(d_C));

  CHECK_CUDA(cudaStreamDestroy(stream));
}

void BenchDnAxSpB2(const int m, const int n, const int k, const int sparsity) {
  int A_size = m * k;
  int B_size = n * k;
  int C_size = m * n;
  int B_nnz = 0;

  float* A = new float[A_size]();
  float* B = new float[B_size]();
  float* B_trans = new float[B_size]();
  float* B_csr_values;
  int *B_csr_col_indices;
  int *B_csr_row_offsets;
  int *B_csr_row_indices;
  float* C = new float[C_size]();
  float* C_cublas = new float[C_size]();

  float* d_A;
  float* d_B_csr_values;
  int *d_B_csr_col_indices;
  int *d_B_csr_row_offsets;
  int *d_B_csr_row_indices;
  float* d_B;
  float* d_C;
  
  cudaStream_t stream;
  
  seqmm::SputnikMM sputnik_spmm = seqmm::SputnikMM();
  seqmm::CuBlasMM<float> cublas_gemm = seqmm::CuBlasMM<float>();

  CHECK_CUDA(cudaStreamCreate(&stream));
  
  CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_C, C_size * sizeof(float)));
    
  printf("[Generate Random Dense Matrix A] - Starting...");
  seqmm::GenDenseMatrix(A, m, k);
  printf("done.\n");

  CHECK_CUDA(cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice));

  printf("[Generate Random Sparse Matrix B] - Starting...");
  seqmm::GenSparseMatrix(B, k, n, sparsity);
  printf("done.\n");

  printf("[Transpose B to B_trans] - Starting...");
  seqmm::Transpose(B, k, n, B_trans);
  printf("done.\n");

  printf("[Convert Dense Matrix B To Sparse] - Starting...");
  seqmm::SparseMatrixGetSize(B_trans, B_size, &B_nnz);
  B_csr_values = new float[B_nnz] ();
  B_csr_col_indices = new int[B_nnz] ();
  B_csr_row_offsets = new int[n + 1] ();
  B_csr_row_indices = new int[n] ();
  seqmm::Dense2Sparse(B_trans, n, k, 
                      B_csr_values, B_csr_col_indices, B_csr_row_offsets);
  std::iota(B_csr_row_indices, B_csr_row_indices + n, 0);
  printf("done.\n");

  CHECK_CUDA(cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&d_B_csr_values, B_nnz * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_values, B_csr_values, B_nnz * sizeof(float),
                        cudaMemcpyHostToDevice));
  
  CHECK_CUDA(cudaMalloc(&d_B_csr_col_indices, B_nnz * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_col_indices, B_csr_col_indices,
                        B_nnz * sizeof(int), cudaMemcpyHostToDevice));
  
  CHECK_CUDA(cudaMalloc(&d_B_csr_row_offsets, (n + 1) * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_row_offsets, B_csr_row_offsets,
                        (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&d_B_csr_row_indices, n * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_row_indices, B_csr_row_indices,
                        n * sizeof(int), cudaMemcpyHostToDevice));

  sputnik_spmm.Init();
  
  int iter = 100;
  printf("[Matrix Multiplication with Sputnik] - Starting...");
  sputnik_spmm.Run(d_A, d_B_csr_values, d_B_csr_col_indices, d_B_csr_row_offsets,
                   d_B_csr_row_indices, B_nnz, m, n, k, d_C, true, stream);
  CUDA_EVENT_START(sputnik, "Sputnik");
  for (int i = 0; i < iter; ++i) {
    sputnik_spmm.Run(d_A, d_B_csr_values, d_B_csr_col_indices, d_B_csr_row_offsets,
                     d_B_csr_row_indices, B_nnz, m, n, k, d_C, true, stream);
  }
  CUDA_EVENT_STOP(sputnik, "Sputnik");
  printf("done.\n");

  CHECK_CUDA(cudaMemcpy(C, d_C, C_size * sizeof(float), cudaMemcpyDeviceToHost));

  cublas_gemm.Init(1.0f, 0.0f, false, false, m, n, k);
  printf("[Matrix Multiplication with CuBLAS] - Starting...");
  cublas_gemm.Run(d_A, d_B, d_C, m, n, k);

  CUDA_EVENT_START(cublas, "CUBLAS");
  for (int i = 0; i < iter; ++i) {
    cublas_gemm.Run(d_A, d_B, d_C, m, n, k);
  }
  CUDA_EVENT_STOP(cublas, "CUBLAS");
  printf("done.\n");
  
  CHECK_CUDA(cudaMemcpy(C_cublas, d_C, C_size * sizeof(float), cudaMemcpyDeviceToHost));
  
  bool res = seqmm::CompareL2fe(C_cublas, C, C_size, (float)1.0e-6f);
  printf("Comparing CUSPARSE Matrix Multiply with CPU results: %s\n",
         (true == res) ? "PASS" : "FAIL");
    
  // Clean up.
  cublas_gemm.Clear();
  sputnik_spmm.Clear();
  
  delete [] A;
  delete [] B;
  delete [] B_trans;
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
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "Usage: ./" << argv[0] << " m k n sparsity" << std::endl;
    exit(-1);
  }
  int m = atoi(argv[1]);
  int k = atoi(argv[2]);
  int n = atoi(argv[3]);
  int sparsity = atoi(argv[4]);

  //BenchSpAxDnB(m, n, k, sparsity);
  BenchDnAxSpB(m, n, k, sparsity);
  BenchDnAxSpB2(m, n, k, sparsity);
  
  return 0;
}
