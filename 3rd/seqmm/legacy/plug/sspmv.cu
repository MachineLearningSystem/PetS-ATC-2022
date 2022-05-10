#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

#include <vector>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <iostream>

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
            std::cout << std::setw(10) << array[j * col + k] - 0 << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "===============" << std::endl;
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

int bench(int m, int n){
    size_t sparsity = 99;

    int iter = 100;

    std::vector<float> A(m * n);
    std::vector<float> B(n);
    random_init<float>(A.data(), A.size(), sparsity);
    random_init<float>(B.data(), B.size(), 0);

    float *A_d, *B_d, *C_d;
    CHECK_CUDA(cudaMalloc(&A_d, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B_d, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C_d, m * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(A_d, A.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d, B.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseDnMatDescr_t matA_dense;
    cusparseDnVecDescr_t vecB, vecC;

    CHECK_CUSPARSE(cusparseCreateDnMat(&matA_dense,
        m, n, n, A_d, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecB, n, B_d, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecC, m, C_d, CUDA_R_32F));

    int *csrRowOffset;
    CHECK_CUDA(cudaMalloc(&csrRowOffset, sizeof(int) * (m + 1)));

    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
        m, n, 0, csrRowOffset, NULL, NULL, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle,
        matA_dense, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));

    void *buffer;
    CHECK_CUDA(cudaMalloc(&buffer, bufferSize));

    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle,
        matA_dense, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer));

    int64_t rows, cols, nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matA, &rows, &cols, &nnz));
    printf("NNZ %ld\n", nnz);

    int *csrColIdx;
    float *csrData;
    CHECK_CUDA(cudaMalloc(&csrColIdx, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&csrData, nnz * sizeof(float)));
    CHECK_CUSPARSE(cusparseCsrSetPointers(matA, csrRowOffset, csrColIdx, csrData));

    CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle,
        matA_dense, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer));

    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseSpMVAlg_t spmv_alg = CUSPARSE_SPMV_ALG_DEFAULT;

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecB,
        &beta, vecC, CUDA_R_32F, spmv_alg, &bufferSize));

    cudaFree(buffer);
    CHECK_CUDA(cudaMalloc(&buffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMV(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecB,
        &beta, vecC, CUDA_R_32F, spmv_alg, buffer));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaProfilerStart());
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iter; ++i) {
        CHECK_CUSPARSE(cusparseSpMV(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecB,
            &beta, vecC, CUDA_R_32F, spmv_alg, buffer));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaProfilerStop());

    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "True Sparity:" << 1 - nnz * 1.0 / ( m * n ) << std::endl;
    std::cout << iter <<  " round cusparseSpMM duration:" << ms << "ms" << std::endl;
    std::cout << "each round on average :" << ms / iter << "ms" << std::endl;

    std::vector<float> C(m);
    CHECK_CUDA(cudaMemcpy(C.data(), C_d, m * sizeof(float), cudaMemcpyDeviceToHost));
    print_matrix<float>(C.data(), 1, 10);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matA_dense));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecB));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecC));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK_CUDA(cudaFree(buffer));
    CHECK_CUDA(cudaFree(A_d));
    CHECK_CUDA(cudaFree(B_d));
    CHECK_CUDA(cudaFree(C_d));

    return 0;
}

int main() {
    int test_case[] = {8192, 1024, 8192, 8192, 24576, 8192, 32768, 8192, 8192, 32768};

    for (int i = 0; i < 5; ++i) {
        bench(test_case[i * 2], test_case[i * 2 + 1]);
    }

    return 0;
}

