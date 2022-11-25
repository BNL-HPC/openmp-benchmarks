#pragma once

#include <iostream>
#include <cuda.h>

template<typename T, int size, int blocksize>
class cuda_bench
{
public:

  cuda_bench();
  ~cuda_bench();
 
  __device__ __global__ void set_array_zero( );

//private:
  T * m_device_array;
  const int m_threads_tot    = size;
  const int m_blocksize      = blocksize;
  const int m_nblocks        = ( m_threads_tot + m_blocksize - 1 ) / m_blocksize;
};
	
template<typename T, int size, int blocksize>
cuda_bench<T, size, blocksize>::cuda_bench() {

  cudaMalloc((void**)&m_device_array, sizeof( T ) * m_threads_tot);	
  /* Allocate array of length N on target */
  if ( m_device_array == NULL ) {
    std::cout << " ERROR: No space left on device." << std::endl;
  }
}

template<typename T, int size, int blocksize>
cuda_bench<T, size, blocksize>::~cuda_bench() {

  cudaFree( m_device_array );
   
}

template<typename T, int size, int blocksize>
__device__ __global__ void cuda_bench<T, size, blocksize>::set_array_zero( ) {
  
  unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;
  m_device_array[tid] = 0.0; 
  return;
}

// Explicitly instantiate only the classes you want to be defined.
//template class cuda_bench<double, 4096, 256>;
//template class cuda_bench<double, 4096, 256>;
//template class cuda_bench<double, 4096, 256>;
