
if(CMAKE_CUDA_COMPILER)
    set(array_init_cuda_sources
      ../src/catch_main.cpp
      array_init_test.cu
      array_init.cu
    )
    add_executable(array_init_cuda_app ${array_init_cuda_sources})
    target_include_directories(array_init_cuda_app PUBLIC ../include)
    set_target_properties(array_init_cuda_app PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
endif()
    
set(array_init_omp_sources
  ../src/catch_main.cpp
  array_init_test.cpp
  array_init.cpp
)

add_executable(array_init_omp_app ${array_init_omp_sources})
target_include_directories(array_init_omp_app PUBLIC ../include)
