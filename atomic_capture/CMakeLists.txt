
if(CMAKE_CUDA_COMPILER)
    set(atomic_capture_cuda_sources
      ../src/catch_main.cpp
      atomic_capture.cu
      atomic_capture_test.cu
    )
    add_executable(atomic_capture_cuda_app ${atomic_capture_cuda_sources})
    target_include_directories(atomic_capture_cuda_app PUBLIC ../include)
    set_target_properties(atomic_capture_cuda_app PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
    target_link_libraries(atomic_capture_cuda_app Catch2WithMain )
endif()

if(CMAKE_HIP_COMPILER)
    set(atomic_capture_hip_sources
      ../src/catch_main.cpp
      atomic_capture.hip
      atomic_capture_test.hip
    )
    add_executable(atomic_capture_hip_app ${atomic_capture_hip_sources})
    target_include_directories(atomic_capture_hip_app PUBLIC ../include)
    target_link_libraries(atomic_capture_hip_app Catch2WithMain )
endif()

set(atomic_capture_omp_sources
  ../src/catch_main.cpp
  ../src/utils.cpp
  atomic_capture_test.cpp
  atomic_capture.cpp
)
add_executable(atomic_capture_omp_app ${atomic_capture_omp_sources})
target_include_directories(atomic_capture_omp_app PUBLIC ../include)
target_link_libraries(atomic_capture_omp_app Catch2WithMain )
