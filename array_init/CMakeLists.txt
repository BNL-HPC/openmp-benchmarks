
if(CMAKE_CUDA_COMPILER)
    set(array_init_cuda_sources
      ../src/catch_main.cpp
      array_init_test.cu
      array_init.cu
    )
    add_executable(array_init_cuda_app ${array_init_cuda_sources})
    target_include_directories(array_init_cuda_app PUBLIC ../include)
    set_target_properties(array_init_cuda_app PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
    target_link_libraries(array_init_cuda_app Catch2WithMain )
endif()

if(CMAKE_HIP_COMPILER)
    set(array_init_hip_sources
      ../src/catch_main.cpp
      array_init_test.hip
      array_init.hip
    )
    add_executable(array_init_hip_app ${array_init_hip_sources})
    target_include_directories(array_init_hip_app PUBLIC ../include)
    target_link_libraries(array_init_hip_app Catch2WithMain )
endif()
 
set(array_init_omp_sources
  ../src/catch_main.cpp
  ../src/utils.cpp
  array_init_test.cpp
  array_init.cpp
)

add_executable(array_init_omp_app ${array_init_omp_sources})
target_include_directories(array_init_omp_app PUBLIC ../include)
target_link_libraries(array_init_omp_app Catch2WithMain )
