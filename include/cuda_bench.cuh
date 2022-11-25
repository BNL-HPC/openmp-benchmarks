#pragma once

#include <iostream>
#include <cuda.h>
#include <catch.hpp>

namespace cuda_bench {

template <typename T>
__global__ void set_to_zero( T* cuda_dev_array, const int N );

template <typename T>
__host__ T* allocate_cuda( const int N );

template <typename T>
__host__ T* test_wrapper ( const int N, const int blocksize );

} // namespace cuda_bench

