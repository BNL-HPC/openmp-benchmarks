#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif

#include <cstdlib> 
#include <ctime> 
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip_bench.hpp>
#include <catch.hpp>

namespace hip_bench {

template double* atomic_capture_wrapper <double> ( const int, const int );
template float*  atomic_capture_wrapper <float>  ( const int, const int );
template int*    atomic_capture_wrapper <int>    ( const int, const int );
	
template <typename T>
__global__ void collect_pos( T* hip_dev_array, T* hip_dev_array_pos, unsigned long* ct, const int N ) {

  //TODO no instance of atomicAdd defined with std::size_t as first arg
  unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;

  if ( tid == 0) ct[0] = 0;

  if ( tid < N ) {
    if ( hip_dev_array[tid] > 0 ) {
      unsigned long count  = atomicAdd( ct, 1 );
      hip_dev_array_pos[count] = hip_dev_array[tid];
      //printf("%f %f %d %d \n", hip_dev_array[tid], hip_dev_array_pos[count], ct[0], count);
    }
  }
   printf("ct %lu ", ct[0]);
}

template <typename T>
__host__ void host_array_initialize ( T* host_array, const int N ) {

  srand(time(0));
  for(int i = 0; i < N; i++){
    host_array[i] = (T)(2*drand48() - 1.0);
  } 

  return;
}

template <>
__host__ void host_array_initialize <int> ( int* host_array, const int N ) {

  srand(time(0));
  for(int i = 0; i < N; i++){
    host_array[i] = rand() % 200 - 100;
  } 

  return;
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

  const int threads_tot = N;
  const int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;
  dim3 blocks(nblocks,1,1);
  dim3 threads(blocksize,1,1);

  T *host_array          = (T *) malloc( N * sizeof( T ) );
  T *host_array_positive = (T *) malloc( N * sizeof( T ) );
 
  host_array_initialize ( host_array, N);
  int host_count = collect_positive_serial_host ( host_array, host_array_positive, N );

  T* hip_dev_array;
  T* hip_dev_array_pos;
  unsigned long* devc_count;
  hipMalloc( (void**) &hip_dev_array, sizeof(T) * N);
  hipMalloc( (void**) &hip_dev_array_pos, sizeof(T) * N);
  hipMalloc( (void**) &devc_count, sizeof( unsigned long ) );
  hipMemcpy(hip_dev_array, host_array, sizeof(T) * N, hipMemcpyHostToDevice);
	  
  //BENCHMARK("HIP Atomic Capture Basic") { collect_pos<<< blocks, threads, 0, 0>>> ( hip_dev_array, hip_dev_array_pos, devc_count, N ); }; 
  BENCHMARK_ADVANCED("HIP Atomic Capture")(Catch::Benchmark::Chronometer meter) {
  //  hipMemset(devc_count, 0, sizeof( int ) );
  //  hipMemset(hip_dev_array_pos, 0, sizeof(T) * N );
  //  meter.measure([hip_dev_array, hip_dev_array_pos, devc_count, N, blocks, threads] 
  //  { hipLaunchKernelGGL( collect_pos, blocks, threads, 0, 0, hip_dev_array, hip_dev_array_pos, devc_count, N ); });
  //  hipDeviceSynchronize() ;
  collect_pos<<<blocks, threads, 0, 0>>> ( hip_dev_array, hip_dev_array_pos, devc_count, N ); 
  //hipDeviceSynchronize() ;
 
  unsigned long *host_copy_count = (unsigned long *) malloc( sizeof( unsigned long ) );
  hipMemcpy( host_copy_count, devc_count, sizeof( unsigned long ), hipMemcpyDeviceToHost);

  T* host_copy_array;
  host_copy_array = (T *) malloc( host_copy_count[0] * sizeof( T ) );
  hipMemcpy(host_copy_array, hip_dev_array_pos, host_copy_count[0] * sizeof( T ), hipMemcpyDeviceToHost);

  CHECK( host_copy_count[0] == host_count );

  /* * * check GPU array parallel with CPU array serial * * */
  T sum = 0.;
  for ( int i = 0; i < host_copy_count[0]; i++ ) {
    //std::cout << i << " " << host_copy_array[i] << " " << host_array_positive[i] << std::endl;  
    sum += host_copy_array[i] - host_array_positive[i];
  }

  CHECK( std::fabs ( (float)sum ) < 0.0001f );
  free( host_copy_array );
  free( host_copy_count );
  };

  free( host_array );
  free( host_array_positive );
  hipFree( hip_dev_array );
  hipFree( hip_dev_array_pos );
  hipFree( devc_count );

  return host_array; 
}

} //namespace hip_bench
