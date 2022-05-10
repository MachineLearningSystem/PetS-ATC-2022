#ifndef SEQMM_GEMM_H_
#define SEQMM_GEMM_H_

void GemmCpu(const float* A, const float* B,
             unsigned int hA, unsigned int wA, unsigned int wB,
             float* C);

int GemmCublas(const float* h_A, const float* h_B,
               int m, int k, int n, bool transA, bool transB,
               int n_iter,
               float* h_C);

#endif  // SEQMM_GEMM_H_
