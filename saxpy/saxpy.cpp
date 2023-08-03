#define CATCH_CONFIG_ENABLE_BENCHMARKING
#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#endif

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <catch.hpp>
#include <openmp_bench.h>
#include <omp.h>

namespace openmp_bench {

template double* saxpy_wrapper <double> ( const std::size_t, const std::size_t );
template float*  saxpy_wrapper <float>  ( const std::size_t, const std::size_t );
template int*    saxpy_wrapper <int>    ( const std::size_t, const std::size_t );

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

template <typename T>
void saxpy_kernel ( T* result, T* data_x_device, T* data_y_device, const T fact, const int N, const int nblocks, const int blocksize ) {

  #pragma omp target is_device_ptr(result, data_y_device, data_x_device)
  #pragma omp teams distribute parallel for num_teams(nblocks) num_threads(blocksize)
  for(int i=0; i<N; i++) {
    //printf( "%f %f %f \n", data_y_device[i] + fact * data_x_device[i], data_y_device[i] , fact * data_x_device[i] );
    result[i] = data_y_device[i] + fact * data_x_device[i];
  }

  return; 
}

template <typename T>
T* saxpy_wrapper ( const std::size_t N, const std::size_t blocksize ) {

  #pragma omp target
  if (omp_is_initial_device ()) {
    std::cout << "Target region being executed on host!! Aborting!!!!" << std::endl;
    abort ();
  }

  const std::size_t threads_tot = N;
  const std::size_t nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;

  const T epsilon = 1.e-12;
  const T fact = 2.0;//initialize_random ( epsilon );
  T* data_x = ( T* ) malloc( sizeof( T ) * N );
  T* data_y = ( T* ) malloc( sizeof( T ) * N );
 
  int m_default_device = omp_get_default_device();
  int m_initial_device = omp_get_initial_device();
  std::size_t m_offset = 0;

  T* data_x_device = (T *) omp_target_alloc( sizeof(T) * N, m_default_device);
  T* data_y_device = (T *) omp_target_alloc( sizeof(T) * N, m_default_device);
  T* result        = (T *) omp_target_alloc( sizeof(T) * N, m_default_device);
  T* result_cpy    = (T *) malloc( sizeof( T ) * N );
  if( data_x_device == NULL or data_y_device == NULL or result == NULL ){
    printf(" ERROR: No space left on device.\n");
    exit(1);
  }
 
  BENCHMARK_ADVANCED("OpenMP saxpy")(Catch::Benchmark::Chronometer meter) {
    for(int i=0; i<N; i++)
    {
      data_x[i] = initialize_random ( epsilon );
      data_y[i] = initialize_random ( epsilon );
    }
   
    if ( omp_target_memcpy( data_x_device, data_x, N * sizeof( T ), 
  			  m_offset, m_offset, m_default_device, m_initial_device ) ) {
         std::cout << "ERROR: copy random numbers from cpu to gpu " << std::endl;
    }
    if ( omp_target_memcpy( data_y_device, data_y, N * sizeof( T ), 
  			  m_offset, m_offset, m_default_device, m_initial_device ) ) {
         std::cout << "ERROR: copy random numbers from cpu to gpu " << std::endl;
    }
  
    //BENCHMARK ("OpenMP saxpy") { return saxpy_kernel ( result, data_x_device, data_y_device, fact, N, nblocks, blocksize); };
    meter.measure( [result, data_x_device, data_y_device, fact, N, nblocks, blocksize] 
    { return saxpy_kernel ( result, data_x_device, data_y_device, fact, N, nblocks, blocksize); });

    if ( omp_target_memcpy( result_cpy, result, N * sizeof( T ),
          		  m_offset, m_offset, m_initial_device, m_default_device ) ) {
         std::cout << "ERROR: copy gpu to cpu " << std::endl;
    }
  
    for(int i = 0; i < N; i++) {
      CHECK ( std::fabs(  result_cpy[i] - ( data_y[i] + fact * data_x[i]  ) ) <= epsilon );
    }
  };
  return data_x; 
}



} //namespace openmp_bench
