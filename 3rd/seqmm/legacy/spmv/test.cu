#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_profiler_api.h>

#include <vector>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <iostream>

#include "half.hpp"

#define NNZ_EPSILON (1e-10)
#define ZERO 0.00

using half_float::half;

#define FP32 1

typedef float fp_type;


template<typename T>
void random_init(T* arr, size_t len, size_t sparsity){
    srand(45678);
    for (size_t i = 0; i < len; ++i) {
        if (rand() % 100 >= sparsity)
            arr[i] = rand() * 1.0 / RAND_MAX * 2.0f - 1.0f;
        else
            arr[i] = 0;
    }
}
template<typename DType>
void print_matrix(const DType *array, int row, int col) {
    // int j, k;
    // std::cout << "-----" << row << " x " << col << "-----" << std::endl;
    // std::cout.precision(4);
    // std::cout.flags(std::ios_base::fixed);
    // for(j = 0; j < row; ++j) {
    //     for(k = 0; k < col; ++k) {
    //         std::cout << std::setw(10) << array[j * col + k] - 0 << "\t";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "===============" << std::endl;
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

template<typename T>
void sparseMatrixGetSize(T* matrix, int size, int* nnz) {
  int _nnz = 0;
  for (int i = 0; i < size; i++) {
    if (abs((float)matrix[i] - ZERO) > NNZ_EPSILON) {
      _nnz++;
    }
  }
  *nnz = _nnz;
}

template<typename T>
void dense2sparse(T* dense_matrix, int row, int column,
                  T* csr_values, 
                  int* csr_col_indices,
                  int* csr_row_offsets) {
  int _nnz = 0;
  csr_row_offsets[0] = 0;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < column; j++ ) {
      if (abs((float)dense_matrix[i*column+j] - ZERO) > NNZ_EPSILON)  {
        csr_values[_nnz] = dense_matrix[i*column+j];
        csr_col_indices[_nnz] = j;
        _nnz++;
      }
    }
    csr_row_offsets[i+1] = _nnz;
  }
}

template<typename T>
int Test(T *dC,
         int m, int k, int n,
         T* dB_csr_values,
         int* dB_col_indices, 
         int* dB_row_offsets, 
         int nnz,
         T* dA,
         bool transA = false,
         bool transB = true) { 
  
  cusparseOperation_t co_transA = transA ? CUSPARSE_OPERATION_TRANSPOSE :
      CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t co_transB = transB ? CUSPARSE_OPERATION_TRANSPOSE :
      CUSPARSE_OPERATION_NON_TRANSPOSE;

  cusparseSpMatDescr_t matB;
  cusparseDnMatDescr_t matA, matC;

  cusparseHandle_t handle = NULL;

  void* dBuffer = nullptr;
  size_t bufferSize;

  float alpha = 1.0f;
  float beta = 0.0f;
  
  int lda = transA ? m : k;
  int ldb = transB ? k : n;
  int ldc = n;

  CHECK_CUSPARSE( cusparseCreate(&handle) );

#if (FP32 == 0)
  // Create dense matrix A
  CHECK_CUSPARSE( cusparseCreateDnMat(&matA, k, m, lda, dA,
                                      CUDA_R_16F, CUSPARSE_ORDER_COL) );
  // Create dense matrix C
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, n, m, ldc, dC,
                                      CUDA_R_16F, CUSPARSE_ORDER_COL) );
  
  CHECK_CUSPARSE( cusparseCreateCsr(&matB, k, n, nnz,
                                    dB_row_offsets, dB_col_indices,
                                    dB_csr_values, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_32I, 
                                    CUSPARSE_INDEX_BASE_ZERO,
                                    CUDA_R_16F) );

  // allocate an external buffer if needed
  CHECK_CUSPARSE( cusparseSpMM_bufferSize(
      handle,
      CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matB, matA, &beta, matC, CUDA_R_16F,
      CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) );
#else
    // Create dense matrix A
  CHECK_CUSPARSE( cusparseCreateDnMat(&matA, k, m, lda, dA,
                                      CUDA_R_32F, CUSPARSE_ORDER_COL) );
  // Create dense matrix C
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, n, m, ldc, dC,
                                      CUDA_R_32F, CUSPARSE_ORDER_COL) );
  
  CHECK_CUSPARSE( cusparseCreateCsr(&matB, k, n, nnz,
                                    dB_row_offsets, dB_col_indices,
                                    dB_csr_values, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_32I, 
                                    CUSPARSE_INDEX_BASE_ZERO,
                                    CUDA_R_32F) );

  // allocate an external buffer if needed
  CHECK_CUSPARSE( cusparseSpMM_bufferSize(
      handle,
      CUSPARSE_OPERATION_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matB, matA, &beta, matC, CUDA_R_32F,
      CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) );
#endif
  
  CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) );

#if (FP32 == 0)
  // Create dense matrix A
  CHECK_CUSPARSE( cusparseCreateDnMat(&matA, k, m, lda, (void *)dA,
                                      CUDA_R_16F, CUSPARSE_ORDER_COL) );
  // Create dense matrix C
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, n, m, ldc, dC,
                                      CUDA_R_16F, CUSPARSE_ORDER_COL) );

  // execute SpMM swammer
  CHECK_CUSPARSE( cusparseSpMM(handle,
                               CUSPARSE_OPERATION_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matB, matA, &beta, matC, CUDA_R_16F,
                               CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) );
#else
    // Create dense matrix A
  CHECK_CUSPARSE( cusparseCreateDnMat(&matA, k, m, lda, (void *)dA,
                                      CUDA_R_32F, CUSPARSE_ORDER_COL) );
  // Create dense matrix C
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, n, m, ldc, dC,
                                      CUDA_R_32F, CUSPARSE_ORDER_COL) );

  // execute SpMM swammer
  CHECK_CUSPARSE( cusparseSpMM(handle,
                               CUSPARSE_OPERATION_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matB, matA, &beta, matC, CUDA_R_32F,
                               CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) );
#endif

  CHECK_CUSPARSE( cusparseDestroySpMat(matB) );
  CHECK_CUSPARSE( cusparseDestroyDnMat(matA) );
  CHECK_CUSPARSE( cusparseDestroyDnMat(matC) );
  CHECK_CUSPARSE( cusparseDestroy(handle) );
      
  // device memory deallocation
  CHECK_CUDA( cudaFree(dBuffer) );

  return EXIT_SUCCESS;
}


int main() {
  // int test_case[] = {8192, 1024, 8192, 8192, 24576, 8192, 32768, 8192, 8192, 32768};

  //  for (int i = 0; i < 5; ++i) {
  //      bench(test_case[i * 2], test_case[i * 2 + 1]);
  //  }

  int m = 100;
  int n = 100;
  int k = 100;
  int sparsity = 50;
  int nnz = 0;

  fp_type* h_A = new fp_type[m * k];
  fp_type* h_B = new fp_type[k * n];
  fp_type* h_C = new fp_type[m * n];

  random_init(h_A, m * k, 0);

  std::cout << "A" << std::endl;
  for (int i = 0; i < 100; ++i) {
    std::cout << (float)h_A[i] << std::endl;
  }
  
  random_init(h_B, k * n, sparsity);

  sparseMatrixGetSize(h_B, k * n, &nnz);

  fp_type* B_values = new fp_type[nnz];
  int* col_indices = new int[nnz];
  int* row_offsets = new int[k + 1];
  
  dense2sparse(h_B, k, n, B_values, col_indices, row_offsets);

  std::cout << "B" << std::endl;
  for (int i = 0; i < 100; ++i) {
    std::cout << (float)B_values[i] << std::endl;
  }

  fp_type* d_A;
  fp_type* d_B_values;
  int* d_col_indices;
  int* d_row_offsets;
  fp_type* d_C;

  CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&d_B_values, nnz * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&d_col_indices, nnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_row_offsets, (k + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(half)));

  CHECK_CUDA(cudaMemcpy(d_A, h_A, m * k * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B_values, B_values, nnz * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_col_indices, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_row_offsets, row_offsets, (k + 1) * sizeof(int), cudaMemcpyHostToDevice));

  Test<fp_type>(d_C, m, k, n, d_B_values, d_col_indices, d_row_offsets, nnz, d_A);

  CHECK_CUDA(cudaMemcpy(h_C, d_C, m * n * sizeof(half), cudaMemcpyDeviceToHost));

  std::cout << "C" << std::endl;
  for (int i = 0; i < 100; ++i) {
    std::cout << (float)h_C[i] << std::endl;
  }

  delete [] h_A;
  delete [] h_B;
  delete [] h_C;
  delete [] B_values;
  delete [] col_indices;
  delete [] row_offsets;

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B_values));
  CHECK_CUDA(cudaFree(d_col_indices));
  CHECK_CUDA(cudaFree(d_row_offsets));
  CHECK_CUDA(cudaFree(d_C));

  return 0;
}


