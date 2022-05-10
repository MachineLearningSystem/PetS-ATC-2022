#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <vector>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

using thrust::device_vector;
using thrust::host_vector;

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
    int j, k;
    std::cout << "-----" << row << " x " << col << "-----" << std::endl;
    std::cout.precision(4);
    std::cout.flags(std::ios_base::fixed);
    for(j = 0; j < row; ++j) {
        for(k = 0; k < col; ++k) {
            float var = array[j * col + k];
            std::cout << std::setw(10) << var << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "===============" << std::endl;
}

void PreProcessData(const char* data_file_path,
                    int k, int n,
                    half* dense_matrix) {
  std::ifstream in(std::string(data_file_path), std::ios::binary | std::ios::in);
  in.read((char *) dense_matrix, sizeof(half) * k * n);
  std::cout << in.gcount() << " bytes read" << std::endl;
  
  // close input file stream
  in.close();
}

bool CompareResults(const half *reference, const half *data,
                    const unsigned int len, const float epsilon) {
  bool result;
  int err_cnt = 0;
  for (unsigned int i = 0; i < len; ++i) {
    if (std::abs((float)reference[i] - (float)data[i]) > epsilon) {
      std::cout << (float)reference[i] << "," << (float)data[i] << std::endl;
      err_cnt++;
    }
  }

  std::cout << "Number of errors: " << err_cnt << std::endl;

  result = (err_cnt == 0);

  return result;
}

int main(int argc, char const *argv[]){
    if (argc != 5) {
        std::cerr << "Usage: ./a.out m k n weight_file_path" << std::endl;
        exit(-1);
    }
    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    const char* weight_file_path = argv[4];
    
    std::vector<half> A(m * k);
    std::vector<half> B(k * n);
    random_init<half>(A.data(), A.size(), 0);
    // random_init<half>(B.data(), B.size(), sparsity);
    PreProcessData(weight_file_path, k, n, B.data());

    device_vector<half> A_cuda(A);
    device_vector<half> B_cuda(B);
    device_vector<half> C_cuda(m * n);
    half* ptr_A = raw_pointer_cast(A_cuda.data());
    half* ptr_B = raw_pointer_cast(B_cuda.data());
    half* ptr_C = raw_pointer_cast(C_cuda.data());
    // gemm
    float alpha = 1.0f;
    float beta = 0.0f;
    bool transA_ = false;
    bool transB_ = true;
    cublasOperation_t transA = transA_ ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = transB_ ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int lda = transA_ ? m : k;
    const int ldb = transB_ ? k : n;
    const int ldc = n;

    // ---------------------------- cublas gemm ------------------------------ //
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    cublasGemmEx(handle, transB, transA, n, m, k, &alpha, ptr_B, CUDA_R_16F, ldb,
            ptr_A, CUDA_R_16F, lda, &beta, ptr_C, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cudaEvent_t start,stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    int iter = 100;
    CHECK_CUDA(cudaEventRecord(start,stream));
    for (int i = 0; i < iter; ++i) {
        cublasGemmEx(handle, transB, transA, n, m, k, &alpha, ptr_B, CUDA_R_16F, ldb,
                ptr_A, CUDA_R_16F, lda, &beta, ptr_C, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    }
    CHECK_CUDA(cudaEventRecord(stop,stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds,start,stop));
    std::cout << "cublasGemmEx duration:" << milliseconds / iter << "ms" << std::endl;
    host_vector<half> C(C_cuda);
    // print_matrix<half>(A.data(), m, k);
    // print_matrix<half>(B.data(), k, n);
    print_matrix<half>(C.data(), 1, 10);
    if (handle) cublasDestroy(handle);
    // ---------------------------- cublas gemm d------------------------------ //
    
    // sparse gemm
    cusparseHandle_t sp_handle = 0;
    cusparseCreate(&sp_handle);
    cusparseSetStream(sp_handle, stream);
    // Perform matrix-matrix multiplication with the CSR-formatted matrix A
     // CUSPARSE APIs
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA, matC, matB_dense;
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, k, m, lda, ptr_A,
                                        CUDA_R_16F, CUSPARSE_ORDER_COL) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB_dense, n, k, ldb, ptr_B,
                                        CUDA_R_16F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, n, m, ldc, ptr_C,
                                        CUDA_R_16F, CUSPARSE_ORDER_COL) )
    // Create sparse matrix B in CSR format
    int* csrRowPtr;
    int* csrColInd;
    half* csrVal;
    cudaMalloc((void **)&csrRowPtr, sizeof(int) * (n + 1));
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, n, k, 0,
                                      csrRowPtr, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )
    //////////////// convert B to CSR
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
                                        sp_handle, matB_dense, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(sp_handle, matB_dense, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp,
                                         &nnz) )

        std::cout << "NNZ=" << nnz << std::endl;

    // allocate CSR column indices and values
    CHECK_CUDA( cudaMalloc((void**) &csrColInd, nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &csrVal,  nnz * sizeof(half)) )
    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(matB, csrRowPtr, csrColInd, csrVal) )
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(sp_handle, matB_dense, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 sp_handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matB, matA, &beta, matC, CUDA_R_16F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    cudaFree(dBuffer);
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(sp_handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matB, matA, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iter; ++i) {
        CHECK_CUSPARSE( cusparseSpMM(sp_handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, matB, matA, &beta, matC, CUDA_R_32F,
                                     CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds,start,stop));
    std::cout << "True Sparity:" << 1 - nnz * 1.0 / ( k * n ) << std::endl;
    std::cout << "cusparseSpMM duration:" << milliseconds / iter << "ms" << std::endl;
    host_vector<half> C_sparse(C_cuda);
    print_matrix<half>(C_sparse.data(), 1, 10);

    bool resCUSPARSE = CompareResults(C_sparse.data(), C.data(), m * n, 1.0e-6f);
    printf("Comparing CUSPARSE Matrix Multiply with CPU results: %s\n", (true == resCUSPARSE) ? "PASS" : "FAIL");
    
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(sp_handle) )
    /////////////////////////////////////////////////////////////////////////////////////////
    std::cout << "M x K x N: " << m << 'x' << k << 'x' << n << std::endl;
    return 0;
}
