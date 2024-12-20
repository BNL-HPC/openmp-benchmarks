#pragma once

#include <iostream>
#include <cuda.h>

namespace cuda_bench {

template <typename T>
__global__ void set_to_zero( T* cuda_dev_array, const int N );

template <typename T>
__host__ T* set_to_zero_wrapper ( const int N, const int blocksize );

template <typename T>
__global__ void collect_pos( T* cuda_dev_array, T* cuda_dev_array_pos, int* ct, const int N );

template <typename T>
__host__ T* atomic_capture_wrapper ( const int N, const int blocksize);

template <typename T>
__host__ T* atomic_update_wrapper ( const int N, const int blocksize);

template <typename T>
__host__ T* saxpy_wrapper ( const int N, const int blocksize);

template <typename T>
__host__ T* reduction_wrapper ( const int N, const int blocksize);

template <typename T>
__host__ T* cublas_wrapper_d ( const int M, const int N, const int K);

template <typename T>
__host__ T*  cublas_wrapper_f ( const int M, const int N, const int K);

template <typename T>
__host__ T* gemm_wrapper ( const int M, const int N, const int K, const int blocksize);


} // namespace cuda_bench

