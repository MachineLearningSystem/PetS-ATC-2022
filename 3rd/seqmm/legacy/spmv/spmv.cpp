#include "spmv.h"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>

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


namespace seqmm {

template<class T> int SpMV<T>::Init(const T* h_mat_A, const T* h_vec_B,
                                    T* h_vec_C, const int m, const int n,
                                    const float alpha, const float beta) {
  m_ = m;
  n_ = n;

  h_mat_A_ = h_mat_A;
  h_vec_B_ = h_vec_B;
  h_vec_C_ = h_vec_C;

  CHECK_CUDA(cudaMalloc(&d_mat_A_, m * n * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_vec_B_, n * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_vec_C_, m * sizeof(T)));
  
  CHECK_CUDA(cudaMemcpy(d_mat_A_, h_mat_A, m * n * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_vec_B_, h_vec_B, n * sizeof(T), cudaMemcpyHostToDevice));
  
  cusparseCreate(&handle_);

  CHECK_CUSPARSE(cusparseCreateDnMat(&mat_A_dense_descr_,
                                     m, n, n, (void*)d_mat_A_, CUDA_R_32F,
                                     CUSPARSE_ORDER_ROW));
  CHECK_CUSPARSE(cusparseCreateDnVec(&vec_B_descr_, n, (void*)d_vec_B_, CUDA_R_32F));
  CHECK_CUSPARSE(cusparseCreateDnVec(&vec_C_descr_, m, (void*)d_vec_C_, CUDA_R_32F));

  CHECK_CUDA(cudaMalloc(&csr_row_offsets_, sizeof(int) * (m + 1)));
  CHECK_CUSPARSE(cusparseCreateCsr(&mat_A_descr_,
                                   m, n, 0, csr_row_offsets_, NULL, NULL, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  
  CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle_,
                                                  mat_A_dense_descr_, mat_A_descr_,
                                                  CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                  &cusparse_buffer_size_));  

  CHECK_CUDA(cudaMalloc(&cusparse_buffer_, cusparse_buffer_size_));

  CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle_,
                                                mat_A_dense_descr_, mat_A_descr_,
                                                CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                cusparse_buffer_));

  int64_t rows, cols;
  CHECK_CUSPARSE(cusparseSpMatGetSize(mat_A_descr_, &rows, &cols, &nnz_));
  printf("NNZ %ld\n", nnz_);
  printf("rows %ld\n", rows);
  printf("cols %ld\n", cols);

  CHECK_CUDA(cudaMalloc(&csr_vals_, nnz_ * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&csr_col_indices_, nnz_ * sizeof(int)));
  CHECK_CUSPARSE(cusparseCsrSetPointers(mat_A_descr_, csr_row_offsets_,
                                        csr_col_indices_, csr_vals_));

  CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle_,
                                               mat_A_dense_descr_, mat_A_descr_,
                                               CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                               cusparse_buffer_));

  alpha_ = alpha;
  beta_ = beta;
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle_,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha_, mat_A_descr_, vec_B_descr_,
                                         &beta_, vec_C_descr_, CUDA_R_32F,
                                         spmv_alg_,
                                         &cusparse_buffer_size_));
  
  cudaFree(cusparse_buffer_);
  CHECK_CUDA(cudaMalloc(&cusparse_buffer_, cusparse_buffer_size_));

  return 0;
}

template<class T> int SpMV<T>::Run(const int iter) {
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaProfilerStart());
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iter; ++i) {
    CHECK_CUSPARSE(cusparseSpMV(handle_,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_,
                                mat_A_descr_, vec_B_descr_,
                                &beta_, vec_C_descr_, CUDA_R_32F, spmv_alg_,
                                cusparse_buffer_));
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaProfilerStop());

  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  
  std::cout << "True Sparity:" << 1 - nnz_ * 1.0 / ( m_ * n_ ) << std::endl;
  std::cout << iter <<  " round cusparseSpMM duration:" << ms << "ms" << std::endl;
  std::cout << "each round on average :" << ms / iter << "ms" << std::endl;

  CHECK_CUDA(cudaMemcpy(h_vec_C_, d_vec_C_, m_ * sizeof(float),
                        cudaMemcpyDeviceToHost));


  return 0;
}

template<class T> int SpMV<T>::Clear() {
  CHECK_CUSPARSE(cusparseDestroyDnMat(mat_A_dense_descr_));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vec_B_descr_));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vec_C_descr_));
  CHECK_CUSPARSE(cusparseDestroySpMat(mat_A_descr_));
  CHECK_CUSPARSE(cusparseDestroy(handle_));

  CHECK_CUDA(cudaFree(cusparse_buffer_));

  CHECK_CUDA(cudaFree(d_mat_A_));
  CHECK_CUDA(cudaFree(d_vec_B_));
  CHECK_CUDA(cudaFree(d_vec_C_));
        
  CHECK_CUDA(cudaFree(csr_vals_));
  CHECK_CUDA(cudaFree(csr_col_indices_));
  CHECK_CUDA(cudaFree(csr_row_offsets_));
}

template int SpMV<float>::Init(const float* d_mat_A, const float* d_vec_B,
                               float* d_vec_C,
                               const int m, const int n,
                               const float alpha = 1.0f,
                               const float beta = 0.0f);
template int SpMV<float>::Run(const int iter = 1);
template int SpMV<float>::Clear();

}  // seqmm
