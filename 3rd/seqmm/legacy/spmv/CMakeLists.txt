set(NAME spmv)
if (CUDA_FOUND)
  cuda_add_library(${NAME} SHARED spmv.cpp ${PROJECT_SOURCE_DIR}/seqmm/utils.cpp
                   ${PROJECT_SOURCE_DIR}/seqmm/gemm/gemm.cpp)
  target_include_directories(${NAME} PUBLIC ${PROJECT_SOURCE_DIR}/seqmm/)
  target_link_libraries(${NAME} ${CUDA_CUBLAS_LIBRARIES})
  target_link_libraries(${NAME} ${CUDA_curand_LIBRARY})
  target_link_libraries(${NAME} ${CUDA_cusparse_LIBRARY})
endif()
