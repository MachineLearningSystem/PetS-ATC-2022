#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

// CUDA runtime
#include <cuda_runtime.h>
#include <cusparse.h>

#include "utils.h"
#include "spmm/spmm.h"

#define ROW_MAJOR 1
#define COLUMN_MAJOR 0

using namespace std;

void SpmmCpu(float *C, int m, int n, const float *A,
             const int* col_indices, const int* row_offsets,
             int nnz, const float *B) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0;
      for (int k = *(row_offsets+i); k < *(row_offsets+i+1); k++) {
        sum += A[k] * B[col_indices[k] * n + j];
      }
      C[i * n + j] = (float)sum;
    }
  }             
}

int SpmmCuSparse(int m, int k, int n, int sparsity, int n_iter) {
  /*--- Step 1: prepare storage for A, B and C. ---*/
  
  // A_num_rows = m, A_num_cols = k
  // B_num_rows = k, B_num_cols = n
  // Because cusparseOrder_t = CUSPARSE_ORDER_ROW, here make ld = num_cols
  int lda = k, ldb = n, ldc = n;
  int A_size = lda * m;
  int B_size = ldb * k;
  int C_size = ldc * m;
  float alpha = 1.0f;
  float beta = 0.0f;
  float *hA = new float[A_size] ();
  float *hB = new float[B_size] ();
  float *hC = new float[C_size] ();

  // set seed for rand()
  srand(4567);
  
  // initialize matrix A, B, C
  GenSparseMatrix(hA, m, k, sparsity);
  RandomInit(hB, B_size);
  RandomInit(hC, C_size);

  /*--- Step 2: sparsify A. ---*/

  // get Matrix A with sparse csr
  int nnz = 0;
  sparseMatrixGetSize(hA, m*k, &nnz);
  float *hA_values = new float[nnz] ();
  int *hA_columns = new int[nnz] ();
  int *hA_csrOffsets = new int[m+1] ();

  dense2sparse(hA, m, k, hA_values, hA_columns, hA_csrOffsets);
  
  /*--- Step 3: call SpmmCpu to get CPU results. ---*/

  float *hC_result = new float[C_size] ();
  SpmmCpu(hC_result, m, n, hA_values, hA_columns, hA_csrOffsets, nnz, hB);

  /*--- Step 4: call cusparse APIs to get GPU results and check results. ---*/

  // sparse gemm
  int *dA_csrOffsets, *dA_columns;
  float *dA_values, *dB, *dC;
  
  checkCudaErrors( cudaMalloc((void**) &dA_csrOffsets, (m + 1) * sizeof(int)));
  checkCudaErrors( cudaMalloc((void**) &dA_columns, nnz * sizeof(int)));
  checkCudaErrors( cudaMalloc((void**) &dA_values,  nnz * sizeof(float)));
  checkCudaErrors( cudaMalloc((void**) &dB,         B_size * sizeof(float)) );
  checkCudaErrors( cudaMalloc((void**) &dC,         C_size * sizeof(float)) );
  checkCudaErrors( cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (m + 1) * sizeof(int),
                              cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dA_columns, hA_columns, nnz * sizeof(int),
                              cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dA_values, hA_values, nnz * sizeof(float),
                              cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dB, hB, B_size * sizeof(float),
                              cudaMemcpyHostToDevice));                           

  cusparseHandle_t     handle = NULL;
  cudaStream_t         stream;
  cudaEvent_t          start,stop;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  CHECK_CUSPARSE( cusparseCreate(&handle) )

  // create and start timer
  printf("Computing result using CUPARSE...");

  cudaStreamCreate(&stream);
  cusparseSetStream(handle, stream);
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  // Create sparse matrix A in CSR format
  CHECK_CUSPARSE( cusparseCreateCsr(&matA, m, k, nnz,
                                    dA_csrOffsets, dA_columns, dA_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
  // Create dense matrix B
  CHECK_CUSPARSE( cusparseCreateDnMat(&matB, k, n, ldb, dB,
                                      CUDA_R_32F, CUSPARSE_ORDER_ROW) );
  // Create dense matrix C
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, m, n, ldc, dC,
                                      CUDA_R_32F, CUSPARSE_ORDER_ROW) );
  // allocate an external buffer if needed
  CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
  checkCudaErrors( cudaMalloc(&dBuffer, bufferSize) );

  // execute SpMM swammer
  CHECK_CUSPARSE( cusparseSpMM(handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                               CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) );
  
  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));
  for (int i = 0; i < n_iter; ++i) {
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) );
  }

  printf("done.\n");

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));
  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute true sparsity
  printf("True Sparsity: %lf\n", 1 - nnz * 1.0 / (m * k));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / n_iter;
  double flopsPerMatrixMul = 2.0 * (double)nnz * (double)n;
  double gigaFlops = (flopsPerMatrixMul * 1.0e-9f)
      / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance: %.2f GFlop/s, Time: %.3f msec, Size: %.0f Ops\n",
      gigaFlops,
      msecPerMatrixMul,
      flopsPerMatrixMul);

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE( cusparseDestroySpMat(matA) );
  CHECK_CUSPARSE( cusparseDestroyDnMat(matB) );
  CHECK_CUSPARSE( cusparseDestroyDnMat(matC) );
  CHECK_CUSPARSE( cusparseDestroy(handle) );

  // device result check
  checkCudaErrors( cudaMemcpy(hC, dC, C_size * sizeof(float), cudaMemcpyDeviceToHost) );
  
  bool resCUSPARSE = CompareL2fe(hC_result, hC, C_size, 1.0e-7f);
  
  // printf("Comparing CUSPARSE Matrix Multiply with CPU results: %s\n", (true == resCUSPARSE) ? "PASS" : "FAIL");
  printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

  // clean up memory
  delete []hA;
  delete []hB;
  delete []hC;
  delete []hA_columns;
  delete []hA_csrOffsets;
  delete []hA_values;
  delete []hC_result;
  
  // device memory deallocation
  checkCudaErrors( cudaFree(dBuffer) );
  checkCudaErrors( cudaFree(dA_csrOffsets) );
  checkCudaErrors( cudaFree(dA_columns) );
  checkCudaErrors( cudaFree(dA_values) );;
  checkCudaErrors( cudaFree(dB) );
  checkCudaErrors( cudaFree(dC) );
  
  return (resCUSPARSE == true) ? EXIT_SUCCESS : EXIT_FAILURE;
}


int SpmmCuSparseWithSparseB(float *C,
                            int m, int k, int n,
                            float* B_csr_values,
                            int* B_csr_col_indices, 
                            int* B_row_offsets, 
                            int nnz,
                            float* A,
                            int dense_layout) {
  int lda = k, ldb = n, ldc = n;
  if (dense_layout == ROW_MAJOR) {
    lda = m;
    ldb = k;
  }
  float alpha = 1.0f;
  float beta = 0.0f;
  int A_size = m * k;
  int C_size = m * n;
  //--------------------------------------------------------------------------
  // Device memory management
  int   *dB_csrOffsets, *dB_columns;
  float *dB_values, *dA, *dC;
  checkCudaErrors( cudaMalloc((void**) &dB_csrOffsets,
                          (k + 1) * sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**) &dB_columns, nnz * sizeof(int))    );
  checkCudaErrors( cudaMalloc((void**) &dB_values,  nnz * sizeof(float))  );
  checkCudaErrors( cudaMalloc((void**) &dA,         A_size * sizeof(float)) );
  checkCudaErrors( cudaMalloc((void**) &dC,         C_size * sizeof(float)) );
  checkCudaErrors( cudaMemcpy(dB_csrOffsets, B_row_offsets,
                        (k + 1) * sizeof(int),
                        cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dB_columns, B_csr_col_indices, nnz * sizeof(int),
                              cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dB_values, B_csr_values, nnz * sizeof(float),
                             cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dA, A, A_size * sizeof(float),
                           cudaMemcpyHostToDevice) );
  // checkCudaErrors( cudaMemcpy(dC, C, C_size * sizeof(float),
  //                          cudaMemcpyHostToDevice) );

  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  // TODO: 目前只实现了A，C为COL_MAJOR 的情况
  cusparseHandle_t     handle = NULL;
  cusparseSpMatDescr_t matB;
  cusparseDnMatDescr_t matA, matC;

  // 计时器
  cudaStream_t         stream;
  cudaEvent_t          start,stop;

  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  CHECK_CUSPARSE( cusparseCreate(&handle) );
  // Create sparse matrix B in CSR format
  CHECK_CUSPARSE( cusparseCreateCsr(&matB, k, n, nnz,
                                    dB_csrOffsets, dB_columns, dB_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );

  // Create dense matrix A
  CHECK_CUSPARSE( cusparseCreateDnMat(&matA, k, m, lda, dA,
                                      CUDA_R_32F, CUSPARSE_ORDER_COL) );
  // Create dense matrix C
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, n, m, ldc, dC,
                                      CUDA_R_32F, CUSPARSE_ORDER_COL) );

  // allocate an external buffer if needed
  CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                handle,
                                CUSPARSE_OPERATION_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matB, matA, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) );
  checkCudaErrors( cudaFree(dBuffer));
  checkCudaErrors( cudaMalloc(&dBuffer, bufferSize) );

  cudaStreamCreate(&stream);
  cusparseSetStream(handle, stream);
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // execute SpMM swammer
  CHECK_CUSPARSE( cusparseSpMM(handle,
                               CUSPARSE_OPERATION_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matB, matA, &beta, matC, CUDA_R_32F,
                               CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) );

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));
  for (int i = 0; i < 10; i++) {
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                CUSPARSE_OPERATION_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matB, matA, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) );
  }

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));
  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  // Compute true sparsity
  printf("True Sparsity: %lf\n", 1 - nnz * 1.0 / (k * n));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / 10;
  double flopsPerMatrixMul = 2.0 * (double)nnz * (double)m;
  double gigaFlops = (flopsPerMatrixMul * 1.0e-9f)
      / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance: %.2f GFlop/s, Time: %.3f msec, Size: %.0f Ops\n",
      gigaFlops,
      msecPerMatrixMul,
      flopsPerMatrixMul);

  
  // device result check
  checkCudaErrors( cudaMemcpy(C, dC, C_size * sizeof(float),
                         cudaMemcpyDeviceToHost) );

  CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
  CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
  CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )

  // device memory deallocation
  checkCudaErrors( cudaFree(dBuffer) );
  checkCudaErrors( cudaFree(dB_csrOffsets) );
  checkCudaErrors( cudaFree(dB_columns) );
  checkCudaErrors( cudaFree(dB_values) );
  checkCudaErrors( cudaFree(dA) );
  checkCudaErrors( cudaFree(dC) );
  return EXIT_SUCCESS;
}


void sparseMatrixGetSize(float* matrix, int size, int* nnz) {
  int _nnz = 0;
  for (int i = 0; i < size; i++) {
    if (abs(matrix[i] - ZERO) > NNZ_EPSILON) {
      _nnz++;
    }
  }
  *nnz = _nnz;
}

void dense2sparse(float* dense_matrix, int row, int column,
                  float* csr_values, 
                  int* csr_col_indices,
                  int* csr_row_offsets) {
  int _nnz = 0;
  csr_row_offsets[0] = 0;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < column; j++ ) {
      if (abs(dense_matrix[i*column+j] - ZERO) > NNZ_EPSILON)  {
        csr_values[_nnz] = dense_matrix[i*column+j];
        csr_col_indices[_nnz] = j;
        _nnz++;
      }
    }
    csr_row_offsets[i+1] = _nnz;
  }
}
