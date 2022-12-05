set(cuda_sources
  ../src/catch_main.cpp
  array_init_cuda_test.cu
  array_init.cu
)

add_executable(cuda_app ${cuda_sources})
target_include_directories(cuda_app PUBLIC ../include)

set(omp_sources
  ../src/catch_main.cpp
  array_init_omp_test.cpp
  array_init.cpp
)

add_executable(omp_app ${omp_sources})
target_include_directories(omp_app PUBLIC ../include)
