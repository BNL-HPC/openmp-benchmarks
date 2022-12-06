#define CATCH_CONFIG_ENABLE_BENCHMARKING
#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#endif

#include<cstdlib>
#include<iostream>
#include<ctime>
#include<cuda.h>
#include<cuda_bench.cuh>
#include<catch.hpp>

namespace cuda_bench {

template <typename T>
T initialize_random ( T epsilon ) {

  if (std::is_same <float, T>::value) { 
    return rand() / static_cast <T> (RAND_MAX);
  }
  if (std::is_same <double, T>::value) {
    return rand() / static_cast <T> (RAND_MAX);
  }
  if (std::is_same <int, T>::value) {
    return rand();
  }
}

//template<typename T>
//__global__ void test_atomic_add ( T* res, T* data, int size ) {
__global__ void test_atomic_add ( float* res, float* data, int size ) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
//  if ( i == 0 ) res[i] = 0.;

  if(i < size) {
    atomicAdd( res, data[i]);
  }
}


TEST_CASE("CUDA Atomic Capture") {
  
  using T   = float;
  T epsilon;

  if (std::is_same <float, T>::value) { 
    epsilon = static_cast <T> (1e-2);	
  }
  if (std::is_same <double, T>::value) {
    epsilon = static_cast <T> (1e-6);	
  }
  if (std::is_same <int, T>::value) {
    epsilon = 0;
  }

  const int N   = 4096;

  T* data = (T*)malloc(sizeof(T) * N);
  for(int i=0; i<N; i++)
  {
    data[i] = initialize_random ( epsilon );
  }

  T res = 0.0;
  for(int i=0; i<N; i++)
    res += data[i];


  T* data_d;
  cudaMalloc((void**)&data_d, sizeof(T) * N);
  cudaMemcpy(data_d, data, sizeof(T) * N, cudaMemcpyHostToDevice);

  T* res_d;
  cudaMalloc((void**)&res_d, sizeof(T) );
  
  test_atomic_add<<<N/512,512>>> ( res_d, data_d, N );  
  //BENCHMARK("CUDA") { return test_atomic_add<<<N/512,512>>> ( res_d, data_d, N ); };
  cudaDeviceSynchronize() ;
 
  T res_h = 0.0;
  cudaMemcpy(&res_h, res_d, sizeof(T), cudaMemcpyDeviceToHost);

  std::cout << res_h << " " << res << std::endl;

}


} //namespace cuda_bench
