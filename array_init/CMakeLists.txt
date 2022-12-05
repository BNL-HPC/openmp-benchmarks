set(array_init_cuda_sources
  ../src/catch_main.cpp
  array_init_test.cu
  array_init.cu
)

add_executable(array_init_cuda_app ${array_init_cuda_sources})
target_include_directories(array_init_cuda_app PUBLIC ../include)

set(array_init_omp_sources
  ../src/catch_main.cpp
  array_init_test.cpp
  array_init.cpp
)

add_executable(array_init_omp_app ${array_init_omp_sources})
target_include_directories(array_init_omp_app PUBLIC ../include)
