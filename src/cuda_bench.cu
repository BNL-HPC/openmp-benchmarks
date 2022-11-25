#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif

#include <cuda_bench.cuh>
#include <catch.hpp>
#include <iostream>
#include <cuda.h>

namespace cuda_bench {

template <typename T>
__global__ void set_to_zero( T* cuda_dev_array, const int N ) {

  unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
  cuda_dev_array[tid] = 0.0; 
  return;
}

template <typename T>
__host__ T* allocate_cuda( const int N ) {
  
  T* ptr;
  cudaMalloc((void**)&ptr, sizeof( T ) * N);
  return ptr;
}

template <typename T>
__host__ T* test_wrapper ( const int N, const int blocksize ) {

  /* Allocate an array of length N */
  T* cuda_dev_array = allocate_cuda <T> ( N ); 
  
  int threads_tot = N;
  int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;
 
  //set_to_zero<<<nblocks, blocksize>>>(cuda_dev_array, N );
  std::cout << N << " " << blocksize << std::endl;
  BENCHMARK("CUDA Array Init") { return set_to_zero<<<nblocks, blocksize>>>( cuda_dev_array, N ); };
  
  T *cuda_host_array = (T *) malloc( N * sizeof( T ) );
  cudaMemcpy(cuda_host_array, cuda_dev_array, N * sizeof( T ), cudaMemcpyDeviceToHost);

  for ( int i = 0; i < N; i++ ) 
       if ( std::fabs ( cuda_host_array[i] ) > 1e-20 ) 
	  std::cout << "!!Problem at i = " << i << std::endl;
  
  cudaFree( cuda_dev_array );

  return cuda_dev_array;
}

} // namespace cuda_bench
