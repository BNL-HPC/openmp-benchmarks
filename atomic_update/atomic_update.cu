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

template double* atomic_update_wrapper <double> ( const int, const int );
template float*  atomic_update_wrapper <float>  ( const int, const int );
template int*    atomic_update_wrapper <int>    ( const int, const int );
	
template <typename T> 
T get_epsilon () { return 1.0e-6; }

template <> double get_epsilon <double> () { return 1.0e-6; }
template <> float  get_epsilon <float>  () { return 1.0e-2; }
template <> int    get_epsilon <int>    () { return 0; }

template <typename T>
T initialize_random ( T epsilon ) {

  if (std::is_same <float, T>::value) { 
    return rand() / static_cast <T> (RAND_MAX);
    //return drand48();
  }
  if (std::is_same <double, T>::value) {
    return rand() / static_cast <T> (RAND_MAX);
  }
  if (std::is_same <int, T>::value) {
    return rand();
  }
}

template<typename T>
__global__ void get_residual ( T* res, T* data, const int size ) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if ( i == 0 ) res[i] = 0.;

  if(i < size) {
    atomicAdd( res, data[i]);
  }
}

template<typename T>
__host__ T* atomic_update_wrapper ( const int N, const int blocksize ) {

  int threads_tot = N;
  int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;

  T epsilon = get_epsilon <T> ();

  T* data = (T*)malloc(sizeof(T) * N);

  for(int i=0; i<N; i++)
  {
    data[i] = initialize_random ( epsilon );
  }

  T res = 0.0;
  for(int i=0; i<N; i++)
    res += data[i];
  res = (double) res / (double) N;

  T* data_d;
  cudaMalloc((void**)&data_d, sizeof(T) * N);
  cudaMemcpy(data_d, data, sizeof(T) * N, cudaMemcpyHostToDevice);

  T* res_d;
  cudaMalloc((void**)&res_d, sizeof(T) );
  
  //get_residual<<<nblocks,blocksize>>> ( res_d, data_d, N );  
  BENCHMARK("CUDA") { return get_residual<<<nblocks, blocksize>>> ( res_d, data_d, N ); };
  cudaDeviceSynchronize() ;
 
  T res_h = 0.0;
  cudaMemcpy(&res_h, res_d, sizeof(T), cudaMemcpyDeviceToHost);
  res_h = (double) res_h / (double) N;

  CHECK ( std::fabs(  res_h - res ) <= epsilon );
//  std::cout << res_h << " " << res << std::endl;

  return data;

}

TEST_CASE("CUDA Atomic Capture") {
 
  using T = float;

  const int N   = 4096;
  const int blocksize = 512;
  float* temp1 = atomic_update_wrapper <float> ( N, blocksize );  


}


} //namespace cuda_bench
