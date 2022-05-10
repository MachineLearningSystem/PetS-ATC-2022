cuda_add_executable(cublas_example cublas_example.cpp)
target_include_directories(cublas_example PUBLIC ${PROJECT_SOURCE_DIR}/seqmm/)
target_link_libraries(cublas_example seqmm)

#add_executable(cusparse_example cusparse_example.cpp)
#target_include_directories(cusparse_example PUBLIC ${PROJECT_SOURCE_DIR}/seqmm/)
#target_link_libraries(cusparse_example seqmm)

#cuda_add_executable(paisparse_example paisparse_example.cpp)
#target_include_directories(paisparse_example PUBLIC ${PROJECT_SOURCE_DIR}/seqmm/)
#target_link_libraries(paisparse_example seqmm)

cuda_add_executable(sputnik_benchmark sputnik_benchmark.cpp)
target_include_directories(sputnik_benchmark PUBLIC ${PROJECT_SOURCE_DIR}/seqmm/)
target_link_libraries(sputnik_benchmark seqmm)

#add_executable(plug_example plug_example.cpp)
#target_include_directories(plug_example PUBLIC ${PROJECT_SOURCE_DIR}/seqmm/)
#target_link_libraries(plug_example seqmm)

#add_executable(plug_prune_example plug_prune_example.cpp)
#target_include_directories(plug_prune_example PUBLIC ${PROJECT_SOURCE_DIR}/seqmm/)
#target_link_libraries(plug_prune_example seqmm)

