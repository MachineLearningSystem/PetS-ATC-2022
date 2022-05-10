#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cuda_profiler_api.h>

#include "cusparse_op.h"
#include "paisparse_op.h"
#include "cublas_op.h"

using namespace std;

template <typename T>
void Bench(const int m, const int n, const int k,
           const char* weight_file_path, const int iter) {
  int matrix_size = n * k;
  int A_size = m * k;
  int nnz = 0;
  int C_size = m * n;
  
  T *weight_matrix = new T[matrix_size] ();
  T* A = new T[A_size] ();
  T* C_cusparse = new T[C_size] ();
  T* C_paisparse = new T[C_size] ();
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
  seqmm::PaiSparseMM<T> paisparse_spmm = seqmm::PaiSparseMM<T>();
  seqmm::CuBlasMM<T> cublas_gemm = seqmm::CuBlasMM<T>();

  printf("[Data Preprocessing] - Starting...");
  seqmm::PreProcessData(weight_file_path, k, n, weight_matrix);
  printf("done.\n");

  printf("[Generate Random Dense Matrix A] - Starting...");
  seqmm::GenDenseMatrix(A, m, k);
  printf("done.\n");

  printf("[Convert Dense Matrix B To Sparse] - Starting...");
  seqmm::SparseMatrixGetSize(weight_matrix, matrix_size, &nnz);
  B_csr_values = new T[nnz] ();
  B_csr_col_indices = new int[nnz] ();
  B_csr_row_offsets = new int[n + 1] ();
  seqmm::Dense2Sparse(weight_matrix, n, k, 
                      B_csr_values, B_csr_col_indices, B_csr_row_offsets);
  printf("done.\n");

  std::cout << "True sparsity: " << 1.0 - (nnz * 1.0) / (n * k) << std::endl;

  CHECK_CUDA(cudaMalloc(&d_A, A_size * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_A, A, A_size * sizeof(T), cudaMemcpyHostToDevice));
  
  CHECK_CUDA(cudaMalloc(&d_B, matrix_size * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_B, weight_matrix, matrix_size * sizeof(T),
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

  printf("[Sparse Matrix Multiply PaiSparse] - Starting...\n");
  paisparse_spmm.Run(d_A, d_B_csr_values, d_B_csr_col_indices, d_B_csr_row_offsets,
                     nnz, m, n, k, false, false, 1.0, 0.0, d_C, stream);
   
  // cudaProfilerStart(); 
  CUDA_EVENT_START(paisparse, "PaiSparse");
  for (int i = 0; i < iter; ++i) {
    paisparse_spmm.Run(d_A, d_B_csr_values, d_B_csr_col_indices, d_B_csr_row_offsets,
                       nnz, m, n, k, false, false, 1.0, 0.0, d_C, stream);
  }
  CUDA_EVENT_STOP(paisparse, "PaiSparse");
  // cudaProfilerStop();
  
  CHECK_CUDA(cudaMemcpy(C_paisparse, d_C, C_size * sizeof(T), cudaMemcpyDeviceToHost));
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

  // For PaiSparse fp16, (epsilon==1.0e-6)->FAIL, (epsilon==1.0e-3)->PASS. 
  bool res_paisparse = seqmm::CompareL2fe(C_cublas, C_paisparse, C_size, (T)1.0e-6f);
  printf("Comparing PAISPARSE Matrix Multiply with CUBLAS results: %s\n",
         (true == res_paisparse) ? "PASS" : "FAIL");
      
  // Clean up.
  cublas_gemm.Clear();
  
  delete [] A;
  delete [] weight_matrix;
  delete [] B_csr_col_indices;
  delete [] B_csr_values;
  delete [] B_csr_row_offsets;
  delete [] C_cublas;
  delete [] C_cusparse;
  delete [] C_paisparse;

  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUSPARSE(cusparseDestroy(handle));
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B_csr_values));
  CHECK_CUDA(cudaFree(d_B_csr_col_indices));
  CHECK_CUDA(cudaFree(d_B_csr_row_offsets));
  CHECK_CUDA(cudaFree(d_C));
}

int main(int argc, char **argv) {
  if (argc != 6) {
    printf("Usage: ./plug_example weight_file_path m k n iter\n");
    exit(-1);
  }
  char* weight_file_path = argv[1];
  int m = atoi(argv[2]);
  int k = atoi(argv[3]);
  int n = atoi(argv[4]);
  int iter = atoi(argv[5]);

  Bench<half>(m, n, k, weight_file_path, iter);
  // Bench<float>(m, n, k, weight_file_path, iter);

  return 0;
}
