
if(CMAKE_CUDA_COMPILER)
    set(saxpy_cuda_sources
      ../src/catch_main.cpp
      saxpy_test.cu
      saxpy.cu
    )
    add_executable(saxpy_cuda_app ${saxpy_cuda_sources})
    target_include_directories(saxpy_cuda_app PUBLIC ../include)
    set_target_properties(saxpy_cuda_app PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
    target_link_libraries(saxpy_cuda_app Catch2WithMain )
endif()

if(CMAKE_HIP_COMPILER)
    set(saxpy_hip_sources
      ../src/catch_main.cpp
      saxpy_test.hip
      saxpy.hip
    )
    add_executable(saxpy_hip_app ${saxpy_hip_sources})
    target_include_directories(saxpy_hip_app PUBLIC ../include)
    target_link_libraries(saxpy_hip_app Catch2WithMain )
endif()

set(saxpy_omp_sources
  ../src/catch_main.cpp
  ../src/utils.cpp
  saxpy_test.cpp
  saxpy.cpp
)
add_executable(saxpy_omp_app ${saxpy_omp_sources})
target_include_directories(saxpy_omp_app PUBLIC ../include)
target_link_libraries(saxpy_omp_app Catch2WithMain )
