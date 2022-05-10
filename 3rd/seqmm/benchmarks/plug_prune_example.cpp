#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cuda_profiler_api.h>

#include "cusparse_op.h"
#include "cublas_op.h"
#include "utils.h"

template <typename T>
void Bench(const int m, const int n, const int k, const int threshold,
           const char* weight_file_path, const int iter) {
  int matrix_size = n * k;
  int A_size = m * k;
  int nnz = 0;
  int C_size = m * n;
  
  T *weight_matrix = new T[matrix_size] ();
  T *pruned_weight_matrix = new T[matrix_size] ();
  T* A = new T[A_size] ();
  T* C_cusparse = new T[C_size] ();
  T* C_cublas = new T[C_size] ();

  T* B_csr_values;
  int *B_csr_col_indices;
  int *B_csr_row_offsets;

  T* d_A;
  T* d_B;
  T* d_B_csr_values;
  int* d_B_csr_col_indices;
  int* d_B_csr_row_offsets;
  T* d_C;

  cudaStream_t stream;
  cusparseHandle_t handle;

  seqmm::CuSparseMM<T> cusparse_spmm = seqmm::CuSparseMM<T>();
  seqmm::CuBlasMM<T> cublas_gemm = seqmm::CuBlasMM<T>();

  printf("[Data Preprocessing] - Starting...");
  seqmm::PreProcessData(weight_file_path, n, k, weight_matrix);
  printf("done.\n");

  printf("[Generate Random Dense Matrix A] - Starting...");
  seqmm::GenDenseMatrix(A, m, k);
  printf("done.\n");

  std::cout << "[Analyze Weight Matrix B] - Starting..." << std::endl;
  seqmm::AnalyzeSparsePattern(weight_matrix, n, k);
  std::cout << "done." << std::endl;

  printf("[Prune Weight Matrix B] - Starting...");
  seqmm::Prune(weight_matrix, n, k, threshold, pruned_weight_matrix);
  std::string path = "pruned_weight_matrix.bin";
  seqmm::WriteMatrixToFile(pruned_weight_matrix, matrix_size * sizeof(T), path);
  printf("done.\n");

  printf("[Convert Dense Matrix B To Sparse] - Starting...");
  seqmm::SparseMatrixGetSize(pruned_weight_matrix, matrix_size, &nnz);
  B_csr_values = new T[nnz] ();
  B_csr_col_indices = new int[nnz] ();
  B_csr_row_offsets = new int[n + 1] ();
  seqmm::Dense2Sparse(pruned_weight_matrix, n, k, 
                      B_csr_values, B_csr_col_indices, B_csr_row_offsets);
  printf("done.\n");

  std::cout << "Non-zero number: " << nnz << std::endl;
  std::cout << "True sparsity: " << 1.0 - (nnz * 1.0) / (n * k) << std::endl;

  CHECK_CUDA(cudaMalloc(&d_A, A_size * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_A, A, A_size * sizeof(T), cudaMemcpyHostToDevice));
  
  CHECK_CUDA(cudaMalloc(&d_B, matrix_size * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_B, pruned_weight_matrix, matrix_size * sizeof(T),
                        cudaMemcpyHostToDevice));
  
  CHECK_CUDA(cudaMalloc(&d_B_csr_values, nnz * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_values, B_csr_values, nnz * sizeof(T),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&d_B_csr_col_indices, nnz * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_col_indices, B_csr_col_indices, nnz * sizeof(int),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&d_B_csr_row_offsets, (n + 1) * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_B_csr_row_offsets, B_csr_row_offsets,
                        (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&d_C, C_size * sizeof(T)));

  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUSPARSE(cusparseCreate(&handle));
  CHECK_CUSPARSE(cusparseSetStream(handle, stream));

  printf("[Sparse Matrix Multiply CuSparse] - Starting...\n");
  cusparse_spmm.Run(d_A, d_B_csr_values, d_B_csr_col_indices, d_B_csr_row_offsets,
                    nnz, m, n, k, false, false, 1.0, 0.0, d_C, handle);
 
  cudaProfilerStart(); 
  CUDA_EVENT_START(cusparse, "CuSparse");
  for (int i = 0; i < iter; ++i) {
    cusparse_spmm.Run(d_A, d_B_csr_values, d_B_csr_col_indices, d_B_csr_row_offsets,
                      nnz, m, n, k, false, false, 1.0, 0.0, d_C, handle);
  }
  CUDA_EVENT_STOP(cusparse, "CuSparse");
  cudaProfilerStop();
  CHECK_CUDA(cudaMemcpy(C_cusparse, d_C, C_size * sizeof(T), cudaMemcpyDeviceToHost));
  printf("done.\n");

  printf("\n[Matrix Multiply CuBlas] - Starting...\n");
  cublas_gemm.Init(1.0f, 0.0f, false, true, m, n, k);
  cublas_gemm.Run(d_A, d_B, d_C, m, n, k);

  CUDA_EVENT_START(cublas, "CUBLAS");
  for (int i = 0; i < iter; ++i) {
    cublas_gemm.Run(d_A, d_B, d_C, m, n, k);
  }
  CUDA_EVENT_STOP(cublas, "CUBLAS");
  
  CHECK_CUDA(cudaMemcpy(C_cublas, d_C, C_size * sizeof(T), cudaMemcpyDeviceToHost));
  printf("done.\n");

  bool res_cusparse = seqmm::CompareL2fe(C_cublas, C_cusparse, C_size, (T)1.0e-6f);
  printf("Comparing CUSPARSE Matrix Multiply with CUBLAS results: %s\n",
         (true == res_cusparse) ? "PASS" : "FAIL");

  // Clean up.
  cusparse_spmm.Clear();
  cublas_gemm.Clear();
  
  delete [] A;
  delete [] weight_matrix;
  delete [] pruned_weight_matrix;
  delete [] B_csr_col_indices;
  delete [] B_csr_values;
  delete [] B_csr_row_offsets;
  delete [] C_cublas;
  delete [] C_cusparse;

  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUSPARSE(cusparseDestroy(handle));
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B_csr_values));
  CHECK_CUDA(cudaFree(d_B_csr_col_indices));
  CHECK_CUDA(cudaFree(d_B_csr_row_offsets));
  CHECK_CUDA(cudaFree(d_C));
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cout << "Usage: " << argv[0] << " weight_file_path m k n threshold iter\n";
    exit(-1);
  }
  char* weight_file_path = argv[1];
  int m = atoi(argv[2]);
  int k = atoi(argv[3]);
  int n = atoi(argv[4]);
  int threshold = atoi(argv[5]);
  int iter = atoi(argv[6]);

  Bench<half>(m, n, k, threshold, weight_file_path, iter);

  return 0;
}
