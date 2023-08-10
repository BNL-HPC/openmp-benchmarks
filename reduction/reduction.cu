#define CATCH_CONFIG_ENABLE_BENCHMARKING
#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#endif

#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bench.cuh>
#include <catch.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

namespace cuda_bench {

template double* reduction_wrapper <double> ( const int, const int );
template float*  reduction_wrapper <float>  ( const int, const int );
template int*    reduction_wrapper <int>    ( const int, const int );
	
template <typename T>
T initialize_random ( T epsilon ) {

  if (std::is_same <float, T>::value) { 
    return 2.0 * (rand() / static_cast <T> (RAND_MAX)) - 1.0;
  }
  if (std::is_same <double, T>::value) {
    return 2.0 * (rand() / static_cast <T> (RAND_MAX)) - 1.0;
  }
  if (std::is_same <int, T>::value) {
    return (rand() % 200) - 100;
  }
}

template<typename T>
__host__ void reduction_kernel ( T* sum_h, T* data_d, T* init_val, const int size ) 
{
  thrust::device_ptr<T> dev_ptr(data_d);
  *sum_h = thrust::reduce(dev_ptr, dev_ptr + size, *init_val, thrust::plus<T>());
  return;
}

template<typename T>
__host__ T* reduction_wrapper ( const int N, const int blocksize ) 
{
  T epsilon = 0.49;

  T* data_h     = (T*)malloc(sizeof(T) * N);
  T* sum        = (T*)malloc(sizeof(T) * 1);
  T* init_val   = (T*)malloc(sizeof(T) * 1);

  T* data_d;
  cudaMalloc((void**)&data_d, sizeof(T) * N);

  BENCHMARK_ADVANCED("CUDA reduction")(Catch::Benchmark::Chronometer meter) {
    for(int i=0; i<N; i++)
    {
      data_h[i] = initialize_random ( epsilon );
    }
    init_val[0] = initialize_random ( epsilon );
    cudaMemcpy(data_d, data_h, sizeof(T) * N, cudaMemcpyHostToDevice);
   
    //BENCHMARK("CUDA") { return reduction_kernel<<<nblocks, blocksize>>> ( result_dev, data_x_dev, data_y_dev, fact, N ); };
    meter.measure( [sum, data_d, init_val, N]
    { return reduction_kernel(sum, data_d, init_val, N ); });	    
    cudaDeviceSynchronize() ;
 
    T correct_sum = init_val[0];
    for(int i = 0; i < N; i++) 
    {
      correct_sum += data_h[i];
    }
    CHECK ( std::fabs( correct_sum - sum[0] ) <= epsilon );
  };

  return sum;
}


} //namespace cuda_bench
