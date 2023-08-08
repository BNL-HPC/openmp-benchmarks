#define CATCH_CONFIG_ENABLE_BENCHMARKING
#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#endif

#include <cstdlib>
#include <iostream>
#include <ctime>
#include <hip/hip_runtime.h>
#include <hip_bench.hpp>
#include <catch.hpp>

namespace hip_bench {

template double* saxpy_wrapper <double> ( const int, const int );
template float*  saxpy_wrapper <float>  ( const int, const int );
template int*    saxpy_wrapper <int>    ( const int, const int );

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
__global__ void saxpy_kernel ( T* result_dev, T* data_x_dev, T* data_y_dev, const T fact, const int size ) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(i < size) {
    result_dev[i] = data_y_dev[i] + fact * data_x_dev[i];
  }
}

template<typename T>
__host__ T* saxpy_wrapper ( const int N, const int blocksize ) {

  const int threads_tot = N ; 
  const int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize ;
  dim3 blocks(nblocks);
  dim3 threads(blocksize);

  T epsilon = 1.e-6;

  T* data_x      = (T*)malloc(sizeof(T) * N);
  T* data_y      = (T*)malloc(sizeof(T) * N);
  T* result_host = (T*)malloc(sizeof(T) * N);

  const T fact = initialize_random ( epsilon );
  T* data_x_dev;
  T* data_y_dev;
  T* result_dev;
  hipMalloc((void**)&data_x_dev, sizeof(T) * N);
  hipMalloc((void**)&data_y_dev, sizeof(T) * N);
  hipMalloc((void**)&result_dev, sizeof(T) * N);

  BENCHMARK_ADVANCED("HIP saxpy")(Catch::Benchmark::Chronometer meter) {
    for(int i=0; i<N; i++)
    {
      data_x[i] = initialize_random ( epsilon );
      data_y[i] = initialize_random ( epsilon );
    }
   
    hipMemcpy(data_x_dev, data_x, sizeof(T) * N, hipMemcpyHostToDevice);
    hipMemcpy(data_y_dev, data_y, sizeof(T) * N, hipMemcpyHostToDevice);
   
    //BENCHMARK("HIP") { return saxpy_kernel<<<nblocks, blocksize>>> ( result_dev, data_x_dev, data_y_dev, fact, N ); };
    //TODO: fix return
    meter.measure( [result_dev, data_x_dev, data_y_dev, fact, N, blocks, threads]
    { saxpy_kernel<<<blocks, threads, 0, 0>>> ( result_dev, data_x_dev, data_y_dev, fact, N ); });	    
    hipDeviceSynchronize() ;
    hipMemcpy(result_host, result_dev, sizeof(T) * N, hipMemcpyDeviceToHost);
  
    for(int i = 0; i < N; i++) {
      CHECK ( std::fabs(  result_host[i] - ( data_y[i] + fact * data_x[i]  ) ) <= epsilon );
    }
  };

  return data_x;

}


} //namespace hip_bench
