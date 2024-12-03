
if(CMAKE_CUDA_COMPILER)
    set(cublas_cuda_sources
      ../src/catch_main.cpp
      cublas_test.cu
      cublas.cu
    )
    add_executable(cublas_cuda_app ${cublas_cuda_sources})
    target_include_directories(cublas_cuda_app PUBLIC ../include)
    target_link_libraries(cublas_cuda_app -lcublas)
    set_target_properties(cublas_cuda_app PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
    target_link_libraries(cublas_cuda_app Catch2WithMain )
endif()

if(CMAKE_CUDA_COMPILER)
    set(gemm_cuda_sources
      ../src/catch_main.cpp
      gemm_test.cu
      gemm.cu
    )
    add_executable(gemm_cuda_app ${gemm_cuda_sources})
    target_include_directories(gemm_cuda_app PUBLIC ../include)
    set_target_properties(gemm_cuda_app PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
    target_link_libraries(gemm_cuda_app Catch2WithMain )
endif()
#
##if(CMAKE_HIP_COMPILER)
##    set(cublas_hip_sources
##      ../src/catch_main.cpp
##      cublas_test.hip
##      cublas.hip
##    )
##    add_executable(cublas_hip_app ${cublas_hip_sources})
##    target_include_directories(cublas_hip_app PUBLIC ../include)
##endif()

set(gemm_omp_sources
  ../src/catch_main.cpp
  ../src/utils.cpp
  gemm_test.cpp
  gemm.cpp
)
add_executable(gemm_omp_app ${gemm_omp_sources})
target_include_directories(gemm_omp_app PUBLIC ../include)
target_link_libraries(gemm_omp_app Catch2WithMain )

