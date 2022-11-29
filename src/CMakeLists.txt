add_executable(main_app main.cpp)
include_directories(main_app PUBLIC ../include)

set(cuda_sources
  catch_main.cpp
  array_init.cu
  cuda_bench.cu
)

add_executable(cuda_app ${cuda_sources})
target_include_directories(cuda_app PUBLIC ../include)

set(omp_sources
  catch_main.cpp
  array_init.cpp
  openmp_bench.cpp
)

add_executable(omp_app ${omp_sources})
target_include_directories(omp_app PUBLIC ../include)
