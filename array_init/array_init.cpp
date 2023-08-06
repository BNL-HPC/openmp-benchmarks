#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif

#include <openmp_bench.h>
#include <catch.hpp> 
#include <iostream>
#include <omp.h>
#include <chrono>

namespace openmp_bench {

template double* set_to_zero_wrapper <double> ( const int, const int );
template float*  set_to_zero_wrapper <float>  ( const int, const int );
template int*    set_to_zero_wrapper <int>    ( const int, const int );

template <typename T>
void set_to_zero( T* device_array, const int N, const int blocksize, const int nblocks ) {

  int i;
  #pragma omp target is_device_ptr ( device_array )            
  #pragma omp teams distribute parallel for num_teams(nblocks) num_threads(blocksize) 
  for(i = 0; i < N; i++) {
    //printf(" num teams = %d, num threads = %d", omp_get_num_teams(), omp_get_num_threads() );
    device_array[i] = 0.;
  }

  return;
}    

template <>
void set_to_zero <int> ( int* device_array, const int N, const int blocksize, const int nblocks ) {

  int i;
  #pragma omp target is_device_ptr ( device_array )            
  #pragma omp teams distribute parallel for num_teams(nblocks) num_threads(blocksize) 
  for(i = 0; i < N; i++) {
    //printf(" num teams = %d, num threads = %d", omp_get_num_teams(), omp_get_num_threads() );
    device_array[i] = 0;
  }

  return;
}    

template <typename T>
T* set_to_zero_wrapper( const int N, const int blocksize ) {

  bool is_target_initial_device = false;	
  #pragma omp target map(tofrom: is_target_initial_device)
  if (omp_is_initial_device ()) {
    printf( "Target region being executed on host!! Aborting!!!! \n");
    is_target_initial_device = true;
  }
  if ( is_target_initial_device )
    std::abort ();

  const int m_default_device = omp_get_default_device();
  const int m_initial_device = omp_get_initial_device();
  const std::size_t m_offset = 0;

  int threads_tot = N;
  int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;

  /* Allocate array of length N on target */
  T *device_array = (T *) omp_target_alloc( N * sizeof( T ), m_default_device);
  if ( device_array == NULL ) {
    std::cout << " ERROR: No space left on device." << std::endl;
  }

//  //auto start = std::chrono::steady_clock::now() ;
//  auto start_omp = omp_get_wtime() ;
//  set_to_zero ( device_array, N, blocksize, nblocks );
//  //std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - start;
//  //std::cout << "chrono time " << elapsed_seconds.count() << std::endl;
//  std::cout << "chrono time " << omp_get_wtime() - start_omp << std::endl;


  /* Init device array with zero */
  BENCHMARK("OpenMP Array Init") { return set_to_zero ( device_array, N, blocksize, nblocks ); };

  /* Copy device array to host for tests */
  T *host_array = (T*) malloc( N * sizeof( T ) );
  if ( omp_target_memcpy( host_array, device_array, N * sizeof( T ),
                                  m_offset, m_offset, m_initial_device, m_default_device ) ) {
    std::cout << "ERROR: array " << std::endl;
  }

  bool test_flag = true;
  for ( int i = 0; i < N; i++ ) {
       if ( std::fabs ( host_array[i] ) > 1e-20 ) 
	  //std::cout << "!!Problem at i = " << i << std::endl;
	  test_flag = false;
  }
  
  //REQUIRE(test_flag == true);
  CHECK(test_flag == true);


  omp_target_free( device_array, m_default_device);
  free( host_array );

  return device_array;
}

} // namespace openmp_bench
