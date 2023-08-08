#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif

#include <cstdlib> 
#include <ctime> 
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bench.cuh>
#include <catch.hpp>
#include <common.hpp>

namespace cuda_bench {

template double* atomic_capture_wrapper <double> ( const int, const int );
template float*  atomic_capture_wrapper <float>  ( const int, const int );
template int*    atomic_capture_wrapper <int>    ( const int, const int );
	
template <typename T>
__global__ void collect_pos( T* cuda_dev_array, T* cuda_dev_array_pos, int* ct, const int N ) {

  //TODO no instance of atomicAdd defined with std::size_t as first arg
  unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;

  if ( tid == 0) ct[0] = 0;

  if ( tid < N ) {
    if ( cuda_dev_array[tid] > 0 ) {
      unsigned int count  = atomicAdd( ct, 1 );
      cuda_dev_array_pos[count] = cuda_dev_array[tid];
      //printf("%f %f %d %d \n", cuda_dev_array[tid], cuda_dev_array_pos[count], ct[0], count);
    }
  }
}


template <typename T>
__host__ int collect_positive_serial_host ( T* host_array, T* host_array_positive, const int N ) {

  int host_ser_count = 0;

  for ( int i = 0; i < N; i++ ) {
      if ( host_array[i] > 0. ) {
        host_array_positive[host_ser_count] = host_array[i];
        host_ser_count++;
      }
  }

  return host_ser_count;	  
}


template <typename T>
__host__ T* atomic_capture_wrapper ( const int N, const int blocksize) {

  int threads_tot = N;
  int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;

  T *host_array          = (T *) malloc( N * sizeof( T ) );
  T *host_array_positive = (T *) malloc( N * sizeof( T ) );
 
  //host_array_initialize ( host_array, N);

  T epsilon = 1e-6;

  srand(time(0));
  for(std::size_t i = 0; i < N; i++){
    host_array[i] = common::initialize_random ( epsilon );
  } 

  int host_count = collect_positive_serial_host ( host_array, host_array_positive, N );

  T* cuda_dev_array;
  T* cuda_dev_array_pos;
  int* devc_count;
  cudaMalloc( (void**) &cuda_dev_array, sizeof(T) * N);
  cudaMalloc( (void**) &cuda_dev_array_pos, sizeof(T) * N);
  cudaMalloc( (void**) &devc_count, sizeof( int ) );
  cudaMemcpy(cuda_dev_array, host_array, sizeof(T) * N, cudaMemcpyHostToDevice);
	  
  //collect_pos <<<nblocks, blocksize>>> ( cuda_dev_array, cuda_dev_array_pos, devc_count, N );
  //BENCHMARK("CUDA Atomic Capture") { return collect_pos <<<nblocks, blocksize>>> ( cuda_dev_array, cuda_dev_array_pos, devc_count, N ); };

  BENCHMARK_ADVANCED("CUDA Atomic Capture")(Catch::Benchmark::Chronometer meter) {
    cudaMemset(devc_count, 0, sizeof( int ) );
    cudaMemset(cuda_dev_array_pos, 0, sizeof(T) * N );
    meter.measure([cuda_dev_array, cuda_dev_array_pos, devc_count, N, nblocks, blocksize] 
    { return collect_pos <<<nblocks, blocksize>>> ( cuda_dev_array, cuda_dev_array_pos, devc_count, N ); });
    cudaDeviceSynchronize() ;
  };
 
  int host_copy_count = 0;
  cudaMemcpy( &host_copy_count, devc_count, sizeof( int ), cudaMemcpyDeviceToHost);

  T* host_copy_array;
  host_copy_array = (T *) malloc( host_copy_count * sizeof( T ) );
  cudaMemcpy(host_copy_array, cuda_dev_array_pos, host_copy_count * sizeof( T ), cudaMemcpyDeviceToHost);

  CHECK( host_copy_count == host_count );

  /* * * check GPU array parallel with CPU array serial * * */
  T sum = 0.;
  for ( int i = 0; i < host_copy_count; i++ ) {
    //std::cout << i << " " << host_copy_array[i] << " " << host_array_positive[i] << std::endl;  
    sum += host_copy_array[i] - host_array_positive[i];
  }

  CHECK( std::fabs ( (float)sum ) < 0.0001f );

  return host_array; 
}

} //namespace cuda_bench
