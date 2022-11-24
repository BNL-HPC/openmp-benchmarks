cmake_minimum_required(VERSION 3.23)

project(OpenMP-target-offload-benchmarks VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Add OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
        ##set(OpenMP_FLAGS "-Xopenmp-target=amdgcn -march=gfx906 -fopenmp -fopenmp-targets=amdgcn")
        ##set(OpenMP_FLAGS "-Xopenmp-target -march=sm_70 -fopenmp -fopenmp-cuda-mode -fopenmp-targets=nvptx64 -fopenmp-target-debug=0x5" )
        set(OpenMP_FLAGS "-Xopenmp-target -march=sm_86 -fopenmp -fopenmp-cuda-mode -fopenmp-targets=nvptx64")
        ##set(OpenMP_FLAGS "-mp=gpu -cuda -Minfo=mp") ## for nvc++
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_FLAGS}")
else()
  message(WARNING "Configuring without OpenMP!")
endif()

include_directories(include)

add_subdirectory(src)
add_subdirectory(tests)
