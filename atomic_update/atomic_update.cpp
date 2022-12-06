#define CATCH_CONFIG_ENABLE_BENCHMARKING
#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#endif

#include<iostream>
#include<cstdlib>
#include<cmath>
#include<catch.hpp>
#include<openmp_bench.h>
#include<omp.h>

namespace openmp_bench {

template <typename T> 
T get_epsilon () { return 1.0e-6; }

template <> double get_epsilon <double> () { return 1.0e-6; }
template <> float  get_epsilon <float>  () { return 1.0e-2; }
template <> int    get_epsilon <int>    () { return 0; }

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

template <typename T>
void get_residual ( T* data_device, T* res_device, const int N ) {

  #pragma omp target is_device_ptr(res_device)
    res_device[0] = 0.;

  #pragma omp target is_device_ptr(res_device, data_device)
  #pragma omp teams distribute parallel for
  for(int i=0; i<N; i++) {
    #pragma omp atomic update
    res_device[0] += data_device[i];
  }

  return; 
}




TEST_CASE("Atomic Update")
{
  using T   = double;
  double epsilon = 1e-6;//get_epsilon ();

  constexpr int N   = 4096;

  T* data = ( T* ) malloc( sizeof( T ) * N );
  for(int i=0; i<N; i++)
  {
    data[i] = initialize_random ( epsilon );
  }
  
  T res = 0.0;
  for(int i = 0; i < N; i++)
    res += data[i];

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

  BENCHMARK ("OpenMP Atomic Update") { return get_residual ( data_device, res_device, N); };

  T host_copy_res = 0.;
  if ( omp_target_memcpy( &host_copy_res, res_device, sizeof( T ),
        		  m_offset, m_offset, m_initial_device, m_default_device ) ) {
       std::cout << "ERROR: copy random numbers from gpu to cpu " << std::endl;
  }

  CHECK ( std::fabs(  host_copy_res - res )  < epsilon );
  //std::cout << host_copy_res << " " << res << std::endl;
}


} //namespace openmp_bench
