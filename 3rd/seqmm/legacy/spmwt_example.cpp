#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include "spmm/spmm.h"
#include "gemm/gemm.h"
#include "utils.h"

using namespace std;

int main(int argc, char **argv) {
  if (argc != 5) {
        printf("Usage: ./spmm weight_file_path m k n\n");
        exit(-1);
  }
  char* weight_file_path = argv[1];
  int m = atoi(argv[2]);
  int k = atoi(argv[3]);
  int n = atoi(argv[4]);

  bool transA = false;
  bool transB = true;

  printf("[Data Preprocessing] - Starting...");
  int matrix_size = k * n;
  float *weight_matrix = new float[matrix_size] ();
  PreProcessData(weight_file_path, k, n, weight_matrix);
  printf("done.\n");

  printf("[Generate Random Dense Matrix A] - Starting...");
  int A_size = m * k;
  float* A = new float[A_size] ();
  GenDenseMatrix(A, m, k);
  printf("done.\n");

  printf("[Transformer Dense Matrix B To Sparse] - Starting...");
  int nnz = 0;
  sparseMatrixGetSize(weight_matrix, k*n, &nnz);
  float* B_csr_values = new float[nnz] ();
  int *B_csr_col_indices = new int[nnz] ();
  int *B_row_offsets = new int[k+1] ();

  dense2sparse(weight_matrix, k, n, 
               B_csr_values, B_csr_col_indices, B_row_offsets);
  printf("Done.\n");

  printf("[Sparse Matrix Multiply CuSparse] - Starting...\n");
  int C_size = m * n;
  float* C = new float[C_size] ();

  SpmmCuSparseWithSparseB(C, m, k, n,
                          B_csr_values,
                          B_csr_col_indices,
                          B_row_offsets,
                          nnz,
                          A,
                          COLUMN_MAJOR);
  printf("Done.\n");

  /*
  printf("[Sparse Matrix Multiply On CPU] - Starting...");
  float* C_cpu_result = new float[C_size] ();
  GemmCpu(A, weight_matrix, m, k, n, C_cpu_result);
  printf("done.\n");
  */

  printf("\n[Matrix Multiply CuBlas] - Starting...\n");
  float* C_cpu_result = new float[C_size] ();
  GemmCublas(A, weight_matrix, m, k, n, transA, transB, 10, C_cpu_result);
  printf("done.\n");

  bool resCUSPARSE = CompareL2fe(C_cpu_result, C, C_size, 1.0e-6f);
  printf("Comparing CUSPARSE Matrix Multiply with CPU results: %s\n",
         (true == resCUSPARSE) ? "PASS" : "FAIL");
      
  // clean up memory
  delete [] A;
  delete [] weight_matrix;
  delete [] B_csr_col_indices;
  delete [] B_csr_values;
  delete [] B_row_offsets;
  delete [] C;
  
  delete [] C_cpu_result;
  
  return 0;
}
