#include <cstdlib> 
#include <cstdio>
#include <ctime> 
#include <iostream>
#include <chrono>
#include <array>
#include <cuda.h>

template <typename T>
__global__ void collect_pos( T* cuda_dev_array, T* cuda_dev_array_pos, int* ct, const int N ) {

  unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;

  if ( tid < N ) { cuda_dev_array_pos[tid] = 0.0; }
  if ( tid == 0) ct[0] = 0;

  if ( tid < N ) {
    if ( cuda_dev_array[tid] > 0 ) {
      unsigned int count  = atomicAdd( ct, 1 );
      cuda_dev_array_pos[count] = cuda_dev_array[tid];
      //printf("%f %f %d %d \n", cuda_dev_array[tid], cuda_dev_array_pos[count], ct[0], count);
    }
  }
}

int main(int argc, char *argv[]){

  using real = double;	
  srand(time(0));
  const int N = 4096*512; 
  
  real *host_array = (real *) malloc( N * sizeof( real ) );
 
  for(int i = 0; i < N; i++){
    host_array[i] = 2*drand48() - 1.0;
  } 

//  std::cout << "Random array:" << std::endl;
//  for ( int i = 0; i < N; i++) std::cout << i << " -> " << host_array[i] << std::endl;
//  std::cout << std::endl;

  for ( int loop = 0; loop < 1; loop++) {
  
  /********** CPU Serial **********/
  real *cpu_array_ser_pos = (real *) malloc( N * sizeof( real ) );

  auto start1  = std::chrono::steady_clock::now();
  int cpu_ser_count = 0;
  for ( int i = 0; i < N; i++ ) {
      if ( host_array[i] > 0. ) {
        cpu_array_ser_pos[cpu_ser_count] = host_array[i];
        cpu_ser_count++;
      }
  }
  std::chrono::duration<double> elapsed_seconds1 = std::chrono::steady_clock::now() - start1;
  std::cout << "CPU Serial -- number of pos rand = " << cpu_ser_count << " in " << elapsed_seconds1.count() <<"s" << std::endl;
  ////for ( int i = 0; i < cpu_ser_count; i++ ) std::cout << cpu_array_ser_pos[i] << "\n" ;
  //std::cout << std::endl;
  /*******************************/

  }  

  /************ CUDA *************/
  int blocksize   = 256;
  int threads_tot = N;
  int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;
  std::cout << "CUDA num blocks = " << nblocks << std::endl;

  real* cuda_dev_array;
  cudaMalloc((void**)&cuda_dev_array, sizeof(real) * N);
  cudaMemcpy(cuda_dev_array, host_array, sizeof(real) * N, cudaMemcpyHostToDevice);

  real* cuda_dev_array_pos;
  cudaMalloc((void**)&cuda_dev_array_pos, sizeof(real) * N);
  
  int* ct;
  cudaMalloc( (void**)&ct, sizeof( int ) );
	  
  collect_pos<<<nblocks, blocksize>>>( cuda_dev_array, cuda_dev_array_pos, ct, N );

  int ct_host = 0;
  cudaMemcpy(&ct_host, ct, sizeof( int ), cudaMemcpyDeviceToHost);

  real* cuda_host_array_pos;
  cuda_host_array_pos = (real *) malloc( ct_host * sizeof( real ) );
  cudaMemcpy(cuda_host_array_pos, cuda_dev_array_pos, ct_host * sizeof( real ), cudaMemcpyDeviceToHost);

  std::cout << "CUDA ct host = " << ct_host << std::endl;
//  for ( int i = 0; i < ct_host; i++ ) std::cout << cuda_host_array_pos[i] << std::endl;

  /*******************************/

  return 0; 
}

