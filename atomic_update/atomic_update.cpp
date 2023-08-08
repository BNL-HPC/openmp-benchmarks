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

template double* atomic_update_wrapper <double> ( const std::size_t, const std::size_t );
template float*  atomic_update_wrapper <float>  ( const std::size_t, const std::size_t );
template int*    atomic_update_wrapper <int>    ( const std::size_t, const std::size_t );

template <typename T> 
T get_epsilon () { return 1.0e-6; }

template <> double get_epsilon <double> () { return 1.0e-6; }
template <> float  get_epsilon <float>  () { return 1.0e-2; }
template <> int    get_epsilon <int>    () { return 0; }

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

template <typename T>
void get_residual ( T* data_device, T* res_device, const int N, const int nblocks, const int blocksize ) {

  // This is not required here anymore; being done in BENCHMARK_ADVANCED	
  // #pragma omp target is_device_ptr(res_device)
  //   res_device[0] = 0.;

  #pragma omp target is_device_ptr(res_device, data_device)
  #pragma omp teams distribute parallel for num_teams(nblocks) num_threads(blocksize)
  for(int i=0; i<N; i++) {
    #pragma omp atomic update
    res_device[0] += data_device[i];
  }

  return; 
}

template <typename T>
T* atomic_update_wrapper ( const std::size_t N, const std::size_t blocksize ) {

  check_target_device ();

  const std::size_t threads_tot = N;
  const std::size_t nblocks      = ( threads_tot + blocksize - 1 ) / blocksize;

  T epsilon = get_epsilon <T> ();

  T* data = ( T* ) malloc( sizeof( T ) * N );
  for(int i=0; i<N; i++)
  {
    data[i] = initialize_random ( epsilon );
  }
  
  T res = 0.0;
  for(int i = 0; i < N; i++)
    res += data[i];
  res = (double) res / (double) N;

  int m_default_device = omp_get_default_device();
  int m_initial_device = omp_get_initial_device();
  std::size_t m_offset = 0;

  T* data_device = (T *) omp_target_alloc( sizeof(T) * N, m_default_device);
  T* res_device  = (T *) omp_target_alloc( sizeof(T), m_default_device);
  if( data_device == NULL ){
    printf(" ERROR: No space left on device.\n");
    exit(1);
  }
 
  if ( omp_target_memcpy( data_device, data, N * sizeof( T ), 
			  m_offset, m_offset, m_default_device, m_initial_device ) ) {
       std::cout << "ERROR: copy random numbers from cpu to gpu " << std::endl;
  }

  //BENCHMARK ("OpenMP Atomic Update") { return get_residual ( data_device, res_device, N, nblocks, blocksize); };

  BENCHMARK_ADVANCED("OpenMP Atomic Update")(Catch::Benchmark::Chronometer meter) {
    #pragma omp target is_device_ptr ( res_device )
      res_device[0] = 0;
    meter.measure([data_device, res_device, N, nblocks, blocksize] 
    { return get_residual ( data_device, res_device, N, nblocks, blocksize); });
  };
 
  T host_copy_res = 0.;
  if ( omp_target_memcpy( &host_copy_res, res_device, sizeof( T ),
        		  m_offset, m_offset, m_initial_device, m_default_device ) ) {
       std::cout << "ERROR: copy random numbers from gpu to cpu " << std::endl;
  }
  host_copy_res = (double) host_copy_res / (double) N;

  CHECK ( std::fabs(  host_copy_res - res ) <= epsilon );
  //std::cout << host_copy_res << " " << res << std::endl;

  return data; 
}



} //namespace openmp_bench
