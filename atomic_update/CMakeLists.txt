set(atomic_update_cuda_sources
  ../src/catch_main.cpp
  #atomic_update_test.cu
  atomic_update.cu
)

add_executable(atomic_update_cuda_app ${atomic_update_cuda_sources})
target_include_directories(atomic_update_cuda_app PUBLIC ../include)

set(atomic_update_omp_sources
  ../src/catch_main.cpp
  atomic_update_test.cpp
  atomic_update.cpp
)

add_executable(atomic_update_omp_app ${atomic_update_omp_sources})
target_include_directories(atomic_update_omp_app PUBLIC ../include)
