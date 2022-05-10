#include <iostream>
#include "cublas_op.h"
#include "cpu_op.h"
#include "utils.h"

template <typename T>
void Bench(const int m, const int n, const int k, const int n_iter) {
  printf("[Generate Random Dense Matrix A] - Starting...");
  int A_size = m * k;
  T* A = new T[A_size] ();
  seqmm::GenDenseMatrix(A, m, k);
  printf("done.\n");
  
  printf("[Generate Random Dense Matrix B] - Starting...");
  int B_size = k * n;
  T *B = new T[B_size] ();
  seqmm::GenDenseMatrix(B, k, n);
  printf("done.\n");

  int C_size = m * n;
  T *d_A, *d_B, *d_C;

  CHECK_CUDA(cudaMalloc((void **) &d_A, A_size * sizeof(T)));
  CHECK_CUDA(cudaMalloc((void **) &d_B, B_size * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_A, A, A_size * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, B_size * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc((void **) &d_C, C_size * sizeof(T)));
  
  printf("[Matrix Multiply CUBLAS] - Starting...\n");
  seqmm::CuBlasMM<T> cublas_gemm = seqmm::CuBlasMM<T>();
  cublas_gemm.Init(1.0f, 0.0f, false, false, m, n, k);
  cublas_gemm.Run(d_A, d_B, d_C, m, n, k);

  float* C = new T[C_size] ();
  CHECK_CUDA(cudaMemcpy(C, d_C, C_size * sizeof(T), cudaMemcpyDeviceToHost));

  printf("[Matrix Multiply CPU] - Starting...\n");
  T* C_golden = new T[C_size] ();
  seqmm::CpuMM<T> cpu_gemm = seqmm::CpuMM<T>();
  cpu_gemm.Init(1.0f, 0.0f, false, false, m, n, k);
  cpu_gemm.Run(A, B, C_golden, m, n, k);

  bool res = seqmm::CompareL2fe(C_golden, C, C_size, 1.0e-6f);
  printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n",
         (true == res) ? "PASS" : "FAIL");

  cublas_gemm.Clear();
  cpu_gemm.Clear();
  
  delete [] A;
  delete [] B;
  delete [] C;
  delete [] C_golden;
  
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
}

int main(int argc, char **argv) {
  if (argc != 5) {
    printf("Usage: ./gemm m k n n_iter\n");
    exit(-1);
  }

  int m = atoi(argv[1]);
  int k = atoi(argv[2]);
  int n = atoi(argv[3]);
  int n_iter = atoi(argv[4]);

  Bench<float>(m, n, k, n_iter);

  return 0;
}
