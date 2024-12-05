#define CATCH_CONFIG_ENABLE_BENCHMARKING
#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#endif

#include <cstdlib>
#include <iostream>
#include <ctime>
#include <hip/hip_runtime.h>
#include <hip_bench.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <common.hpp>

namespace hip_bench {

template double* atomic_update_wrapper <double> ( const int, const int );
template float*  atomic_update_wrapper <float>  ( const int, const int );
template int*    atomic_update_wrapper <int>    ( const int, const int );
	
template<typename T>
__global__ void get_residual ( T* res, T* data, const int size ) {

  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if ( i == 0 ) res[i] = 0.;

  if(i < size) {
    atomicAdd( res, data[i]);
  }
}

template<typename T>
__host__ T* atomic_update_wrapper ( const int N, const int blocksize ) {

  const int threads_tot = N;
  const int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;
  dim3 blocks(nblocks);
  dim3 threads(blocksize);

  T epsilon = common::get_epsilon <T> ();

  T* data = (T*)malloc(sizeof(T) * N);

  for(int i=0; i<N; i++) {
    data[i] = common::initialize_random ( epsilon );
  }

  T res = 0.0;
  for(int i=0; i<N; i++)
    res += data[i];
  res = (double) res / (double) N;

  T* data_d;
  hipMalloc((void**)&data_d, sizeof(T) * N);
  hipMemcpy(data_d, data, sizeof(T) * N, hipMemcpyHostToDevice);

  T* res_d;
  hipMalloc((void**)&res_d, sizeof(T) );
  
  std::cout << __FILE__ << " " << __LINE__ <<std::endl;
  BENCHMARK_ADVANCED("HIP Atomic Update")(Catch::Benchmark::Chronometer meter) {
  std::cout << __FILE__ << " " << __LINE__ <<std::endl;
    hipMemset( res_d, 0, sizeof(T) );
  std::cout << __FILE__ << " " << __LINE__ <<std::endl;
    meter.measure([res_d, data_d, N, blocks, threads] 
      { get_residual<<<blocks, threads>>> ( res_d, data_d, N ); }
    );
    hipDeviceSynchronize() ;
  };
 
  T res_h = 0.0;
  hipMemcpy(&res_h, res_d, sizeof(T), hipMemcpyDeviceToHost);
  res_h = (double) res_h / (double) N;

  CHECK ( std::fabs(  res_h - res ) <= epsilon );
//  std::cout << res_h << " " << res << std::endl;
  std::cout << __FILE__ << " " << __LINE__ <<std::endl;

  return data;

}


} //namespace hip_bench
