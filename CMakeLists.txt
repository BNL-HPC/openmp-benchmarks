cmake_minimum_required(VERSION 3.21)

project(OpenMP-target-offload-benchmarks VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

##add_subdirectory(Catch2)

if(NOT DEFINED Catch2_ROOT)
  message("-- Download and configure Catch2 from GitHub")
  include(FetchContent)
  set(FETCHCONTENT_QUIET OFF)

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        origin/devel
    )

  FetchContent_Populate(Catch2)
  add_subdirectory(${CMAKE_BINARY_DIR}/_deps/catch2-src)
else()
  add_subdirectory(${Catch2_ROOT})
endif()


include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    set(CMAKE_CUDA_STANDARD 14)
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    add_definitions(-DUSE_HIP)
    set(CMAKE_HIP_STANDARD 14)
endif()

# Add OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -fopenmp-cuda-mode")
  set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -foffload-lto")
  set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -fopenmp-assume-no-thread-state")
  #set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -fopenmp-assume-no-nested-parallelism")
  #set(OpenMP_OPT_FLAGS "${OpenMP_OPT_FLAGS} -maxrregcount=64")
  set(OpenMP_OPT_RMRKS "-Rpass=openmp-opt -Rpass-analysis=openmp-opt -Rpass-missed=openmp-opt " )
  #set(OpenMP_DBG_FLAGS "-fopenmp-target-debug=0x5" )
  set(OpenMP_FLAGS "-fopenmp --offload-arch=sm_86 ") 
  #set(OpenMP_FLAGS "-fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a ") 
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_FLAGS} ${OpenMP_OPT_FLAGS} ${OpenMP_OPT_RMRKS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_FLAGS} ${OpenMP_OPT_FLAGS} ${OpenMP_OPT_RMRKS}")
else()
  message(WARNING "Configuring without OpenMP!")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(include)

add_subdirectory(saxpy)
#add_subdirectory(reduction)
add_subdirectory(array_init)
#add_subdirectory(atomic_update)
#add_subdirectory(atomic_capture)
##add_subdirectory(atomic_reduce)
