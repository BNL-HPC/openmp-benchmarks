
if(CMAKE_CUDA_COMPILER)
    set(atomic_update_cuda_sources
      ../src/catch_main.cpp
      atomic_update_test.cu
      atomic_update.cu
    )
    add_executable(atomic_update_cuda_app ${atomic_update_cuda_sources})
    target_include_directories(atomic_update_cuda_app PUBLIC ../include)
    set_target_properties(atomic_update_cuda_app PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
    target_link_libraries(atomic_update_cuda_app Catch2WithMain )
endif()

if(CMAKE_HIP_COMPILER)
    set(atomic_update_hip_sources
      ../src/catch_main.cpp
      atomic_update_test.hip
      atomic_update.hip
    )
    add_executable(atomic_update_hip_app ${atomic_update_hip_sources})
    target_include_directories(atomic_update_hip_app PUBLIC ../include)
    target_link_libraries(atomic_update_hip_app Catch2WithMain )
endif()

set(atomic_update_omp_sources
  ../src/catch_main.cpp
  ../src/utils.cpp
  atomic_update_test.cpp
  atomic_update.cpp
)
add_executable(atomic_update_omp_app ${atomic_update_omp_sources})
target_include_directories(atomic_update_omp_app PUBLIC ../include)
target_link_libraries(atomic_update_omp_app Catch2WithMain )

