#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif

#include <catch.hpp>
#include <hip/hip_runtime.h>
#include <hip_bench.hpp>
#include <iostream>
#include <cstdlib>

namespace hip_bench {

template double* set_to_zero_wrapper <double> ( const int, const int );
template float*  set_to_zero_wrapper <float>  ( const int, const int );
template int*    set_to_zero_wrapper <int>    ( const int, const int );
	
template <typename T>
__global__ void set_to_zero( T* hip_dev_array, const int N ) {

  unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
  hip_dev_array[tid] = 0.0; 
  return;
}

template <>
__global__ void set_to_zero <int> ( int* hip_dev_array, const int N ) {

  unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
  hip_dev_array[tid] = 0; 
  return;
}

template <typename T>
__host__ T* set_to_zero_wrapper ( const int N, const int blocksize ) {

  /* Allocate an array of length N on host */
  T *data = (T *) malloc( N * sizeof( T ) );
  for(int i=0; i<N; i++)
  {
    data[i] = drand48();
  }

  /* Allocate an array of length N on device */
  T* hip_dev_array; 
  hipMalloc((void**)&hip_dev_array, sizeof( T ) * N);
  
  hipMemcpy(hip_dev_array, data, N * sizeof( T ), hipMemcpyHostToDevice);

  const int threads_tot = N;
  const int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;
  dim3 blocks(nblocks,1,1);
  dim3 threads(blocksize,1,1);

  BENCHMARK("HIP Array Init") { set_to_zero<<<blocks, threads, 0, 0>>> ( hip_dev_array, N ); };
  
  T *hip_host_array = (T *) malloc( N * sizeof( T ) );
  hipMemcpy(hip_host_array, hip_dev_array, N * sizeof( T ), hipMemcpyDeviceToHost);

  bool test_flag = true;
  for ( int i = 0; i < N; i++ ) {
       if ( std::fabs ( hip_host_array[i] ) > 1e-20 ) 
	  //std::cout << "!!Problem at i = " << i << std::endl;
	  test_flag = false;
  }
  
  //REQUIRE(test_flag == true);
  CHECK(test_flag == true);

  hipFree( hip_dev_array );
  free( hip_host_array);

  return hip_dev_array;
}

} // namespace hip_bench
