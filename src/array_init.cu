#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif

#include <cuda_bench.cuh>
#include <catch.hpp> 
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <chrono>
#include <array>
#include <vector>
#include <cmath>
#include <cuda.h>

TEST_CASE("CUDA test"){

  using real = double;	
  srand(time(0));
  const int N = 4096; 
  
  real *cuda_host_array = (real *) malloc( N * sizeof( real ) );
 
  for(int i = 0; i < N; i++){
    cuda_host_array[i] = 12.3;
  } 

  int blocksize   = 512;
  int threads_tot = N;
  int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;
  std::cout << "CUDA num blocks = " << nblocks << std::endl;

  /* Allocate an array of length N */
  real* cuda_dev_array;
  cudaMalloc((void**)&cuda_dev_array, sizeof(real) * N);
  /* Copy array from host to device */
  cudaMemcpy(cuda_dev_array, cuda_host_array, sizeof(real) * N, cudaMemcpyHostToDevice);

  //BENCHMARK("CUDA Array Init") { return set_to_zero<<<nblocks, blocksize>>>( ); };

  cudaMemcpy(cuda_host_array, cuda_dev_array, N * sizeof( real ), cudaMemcpyDeviceToHost);

  for ( int i = 0; i < N; i++ ) 
       if ( std::fabs ( cuda_host_array[i] ) > 1e-20 ) 
	  std::cout << "!!Problem at i = " << i << std::endl;

}

