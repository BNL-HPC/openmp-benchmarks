#pragma once

#include <iostream>
#include <cuda.h>
#include <catch.hpp>

namespace cuda_bench {

template <typename T>
__global__ void set_to_zero( T* cuda_dev_array, const int N );

template <typename T>
__host__ T* set_to_zero_wrapper ( const int N, const int blocksize );

template <typename T>
__global__ void collect_pos( T* cuda_dev_array, T* cuda_dev_array_pos, int* ct, const int N );

template <typename T>
__host__ void host_array_initialize ( T* host_array, const int N );

template <typename T>
__host__ int collect_positive_serial_host ( T* host_array, T* host_array_positive, const int N );

template <typename T>
__host__ T* atomic_capture_wrapper ( const int N, const int blocksize);

template <typename T>
__host__ T* atomic_update_wrapper ( const int N, const int blocksize);




} // namespace cuda_bench

