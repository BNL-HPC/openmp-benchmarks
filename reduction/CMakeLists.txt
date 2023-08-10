
if(CMAKE_CUDA_COMPILER)
    set(reduction_cuda_sources
      ../src/catch_main.cpp
      reduction_test.cu
      reduction.cu
    )
    add_executable(reduction_cuda_app ${reduction_cuda_sources})
    target_include_directories(reduction_cuda_app PUBLIC ../include)
    set_target_properties(reduction_cuda_app PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
endif()

#if(CMAKE_HIP_COMPILER)
#    set(reduction_hip_sources
#      ../src/catch_main.cpp
#      reduction_test.hip
#      reduction.hip
#    )
#    add_executable(reduction_hip_app ${reduction_hip_sources})
#    target_include_directories(reduction_hip_app PUBLIC ../include)
#endif()

set(reduction_omp_sources
  ../src/catch_main.cpp
  reduction_test.cpp
  reduction.cpp
)
add_executable(reduction_omp_app ${reduction_omp_sources})
target_include_directories(reduction_omp_app PUBLIC ../include)

