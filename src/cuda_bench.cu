#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif

#include <catch.hpp>
#include <cuda_bench.cuh>
#include <iostream>
#include <cuda.h>

namespace cuda_bench {

template double* set_to_zero_wrapper <double> ( const int, const int );
template float*  set_to_zero_wrapper <float>  ( const int, const int );
template int*    set_to_zero_wrapper <int>    ( const int, const int );
	
template <typename T>
__global__ void set_to_zero( T* cuda_dev_array, const int N ) {

  unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
  cuda_dev_array[tid] = 0.0; 
  return;
}

template <>
__global__ void set_to_zero <int> ( int* cuda_dev_array, const int N ) {

  unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
  cuda_dev_array[tid] = 0; 
  return;
}

template <typename T>
__host__ T* set_to_zero_wrapper ( const int N, const int blocksize ) {

  /* Allocate an array of length N */
  T* cuda_dev_array; 
  cudaMalloc((void**)&cuda_dev_array, sizeof( T ) * N);
  
  int threads_tot = N;
  int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;

  //set_to_zero<<<nblocks, blocksize>>>(cuda_dev_array, N );
  BENCHMARK("CUDA Array Init") { return set_to_zero<<<nblocks, blocksize>>>( cuda_dev_array, N ); };
  
  T *cuda_host_array = (T *) malloc( N * sizeof( T ) );
  cudaMemcpy(cuda_host_array, cuda_dev_array, N * sizeof( T ), cudaMemcpyDeviceToHost);

  bool test_flag = true;
  for ( int i = 0; i < N; i++ ) {
       if ( std::fabs ( cuda_host_array[i] ) > 1e-20 ) 
	  //std::cout << "!!Problem at i = " << i << std::endl;
	  test_flag = false;
  }
  
  //REQUIRE(test_flag == true);
  CHECK(test_flag == true);

  cudaFree( cuda_dev_array );
  free( cuda_host_array);

  return cuda_dev_array;
}

} // namespace cuda_bench
