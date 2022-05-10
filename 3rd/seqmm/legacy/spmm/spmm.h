#ifndef SEQMM_SPMM_H_
#define SEQMM_SPMM_H_

#include <cstdlib>

#ifndef COLUMN_MAJOR
#define ROW_MAJOR 1
#define COLUMN_MAJOR 0
#endif

#define NNZ_EPSILON (1e-10)
#define ZERO 0.00

/**
 * count the non-zero element in matrix.
 * input
 * - matrix: a matrix which would be count non-zero element
 * - size: matrix's size (row * column)
 * output:
 * - nnz: number of non-zero element in matrix
 */
void sparseMatrixGetSize(float* matrix, int size, int* nnz);

/**
 * Extract non-zero elements from original dense matrix, and store them
 * with CSR format.
 * input
 *  - dense_matrx: original dense matrix
 *  - row, column: dimensions of dense_matrix
 * output:
 *  - csr_values: output storage for non-zero elements in sparse_matrix
 *  - col_indices, row_offsets: auxiliary data structures for element indexing
 */
void dense2sparse(float* dense_matrix, int row, int column,
                  float* csr_values, 
                  int* csr_col_indices,
                  int* csr_row_offsets);

void SpmmCpu(float *C,
             int m, int n,
             const float *A,
             const int* col_indices, const int* row_offsets,
             int nnz,
             const float *B);

int SpmmCuSparse(int m, int k, int n, int sparcity, int n_iter);


/**
 * Calculate sparse matrix multiplication based on GPU. 
 * Calculate A × B = C, where A is a sparse matrix with CSR format.
 * input:
 * - m: row dimension of matrix A
 * - k: column dimension of matrix A and row dimension of matrix B
 * - n: column dimension of matrix B
 * - A_csr_value: non-zero elements of a sparse matrix A
 * - A_csr_col_indices: column indices of A_csr_value
 * - A_row_offsets: row offsets of A_csr_value
 * - nnz: number of non-zero elements in A
 * - B: a dense matrix
 * dense_layout: dense matrix layout. It can be ROW_MAJOR or COLUMN_MAJOR.
 * output:
 * - C: a dense matrix C obtained by multiplying point A by B
 * -return:
 * - 0: success
 * - 1: failure
 */
int SpmmCuSparseWithSparseA(float *C,
                            int m, int k, int n,
                            float* A_csr_values,
                            int* A_csr_col_indices, 
                            int* A_row_offsets, 
                            int* nnz,
                            float* B,
                            int dense_layout);


/**
 * Calculate sparse matrix multiplication based on GPU. 
 * Calculate A × B = C, where B is a sparse matrix with CSR format.
 * input:
 * - m: row dimension of matrix A
 * - k: column dimension of matrix A and row dimension of matrix B
 * - n: column dimension of matrix B
 * - B_csr_value: non-zero elements of a sparse matrix B
 * - B_csr_col_indices: column indices of B_csr_value
 * - B_row_offsets: row offsets of B_csr_value
 * - nnz: number of non-zero elements in B
 * - A: a dense matrix
 * dense_layout: dense matrix layout. It can be ROW_MAJOR or COLUMN_MAJOR.
 * output:
 * - C: a dense matrix C obtained by multiplying point A by B
 * * -return:
 * - 0: success
 * - 1: failure
 */
int SpmmCuSparseWithSparseB(float *C,
                            int m, int k, int n,
                            float* B_csr_values,
                            int* B_csr_col_indices, 
                            int* B_row_offsets, 
                            int nnz,
                            float* A,
                            int dense_layout);


#endif  // SEQMM_SPMM_H_
