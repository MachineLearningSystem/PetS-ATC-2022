#ifndef SEQMM_SPMV_H_
#define SEQMM_SPMV_H_

#include <stdio.h>
#include <cusparse_v2.h>

namespace seqmm {

template<class T>
class SpMV {
 public:
  int Init(const T* d_mat_A, const T* d_vec_B, T* d_vec_C,
           const int m, const int n,
           const float alpha = 1.0f,
           const float beta = 0.0f);

  int Run(const int iter = 1);
  
  int Clear();
  
 private:
  int m_, n_;
  int64_t nnz_;
  
  float alpha_, beta_;
  
  cusparseHandle_t handle_;
  cusparseDnMatDescr_t mat_A_dense_descr_;
  cusparseDnVecDescr_t vec_B_descr_, vec_C_descr_;
  cusparseSpMatDescr_t mat_A_descr_;

  const T* h_mat_A_;
  const T* h_vec_B_;
  T* h_vec_C_;
  
  T* d_mat_A_;
  T* d_vec_B_;
  T* d_vec_C_;

  T* csr_vals_;
  int* csr_col_indices_;
  int* csr_row_offsets_;
  
  void* cusparse_buffer_;

  size_t cusparse_buffer_size_ = 0;

  cusparseSpMVAlg_t spmv_alg_ = CUSPARSE_SPMV_ALG_DEFAULT;
};  // class SpMV

}  // namespace seqmm

#endif  // SEQMM_SPMV_M_
