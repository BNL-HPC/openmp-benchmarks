#set(cuda_sources
#  ../src/catch_main.cpp
#  array_init_test.cu
#  array_init.cu
#)

#add_executable(cuda_app ${cuda_sources})
#target_include_directories(cuda_app PUBLIC ../include)

set(atomic_capture_omp_sources
  ../src/catch_main.cpp
  atomic_capture_test.cpp
  atomic_capture.cpp
)

add_executable(atomic_capture_omp_app ${atomic_capture_omp_sources})
target_include_directories(atomic_capture_omp_app PUBLIC ../include)
