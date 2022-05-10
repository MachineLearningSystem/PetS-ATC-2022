#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>

#include "cusparse_op.h"
#include "utils.h"

using namespace std;

int main(int argc, char **argv) {
  if (argc != 4) {
        printf("Usage: ./spmv_example m n sparsity\n");
        exit(-1);
  }
  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  int sparsity = atoi(argv[3]);
  
  printf("[Generate Random Sparse Matrix A] - Starting...");
  int A_size = m * n;
  float* A = new float[A_size] ();
  GenSparseMatrix(A, m, n, sparsity);
  printf("done.\n");
  
  printf("[Generate Random Desne Matrix B] - Starting...");
  int B_size = n * 1;
  float *B = new float[B_size] ();
  GenDenseMatrix(B, n, 1);
  printf("done.\n");

  printf("[Sparse Matrix Vector Multiply CuSparse] - Starting...\n");
  int C_size = m;
  float* C = new float[C_size] ();

  seqmm::CuSparseMV<float> spmv = seqmm::CuSparseMV<float>();
  spmv.Init(A, B, C, m, n);
  spmv.Run(10);
  spmv.Clear();

  printf("Done.\n");

  // clean up memory
  delete [] A;
  delete [] B;
  delete [] C;
  
  return 0;
}
