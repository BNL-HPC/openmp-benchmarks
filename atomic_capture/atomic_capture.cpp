#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif

#include <cstdlib> 
#include <ctime> 
#include <iostream>
#include <openmp_bench.h>
#include <catch.hpp>
#include <common.hpp>
#include <omp.h>

namespace openmp_bench {

template double* atomic_capture_wrapper <double> ( const std::size_t, const std::size_t );
template float*  atomic_capture_wrapper <float>  ( const std::size_t, const std::size_t );
template int*    atomic_capture_wrapper <int>    ( const std::size_t, const std::size_t );


template <typename T>
void collect_positive_devc ( T* devc_array, T* devc_array_positive, std::size_t* devc_count, const std::size_t N, 
		                  const std::size_t nblocks, const std::size_t blocksize ) {

  //Order of positive rands in the output array not preserved

  #pragma omp target is_device_ptr ( devc_count )
  devc_count[0] = 0;
 
  #pragma omp target is_device_ptr ( devc_count, devc_array, devc_array_positive ) 
  #pragma omp teams distribute parallel for num_teams(nblocks) num_threads(blocksize)
  for ( std::size_t i = 0; i < N; i++ ) {

      if ( devc_array[i] > 0. ) {
	std::size_t temp;
	#pragma omp atomic capture
	temp = devc_count[0]++;
	devc_array_positive[temp] = devc_array[i]; 
	//printf( "index=%lu positive rand=%f, %f count=%lu, %lu \n", i, devc_array[i], devc_array_positive[temp], devc_count[0], temp);
      }
  }
 
  return ;	  
}

template <typename T>
std::size_t collect_positive_serial_host ( T* host_array, T* host_array_positive, const std::size_t N ) {

  std::size_t host_ser_count = 0;

  for ( std::size_t i = 0; i < N; i++ ) {
      if ( host_array[i] > 0. ) {
        host_array_positive[host_ser_count] = host_array[i];
        host_ser_count++;
      }
  }

  return host_ser_count;	  
}

template <typename T>
T* atomic_capture_wrapper ( const std::size_t N, const std::size_t blocksize ) {

  check_target_device ();

  const std::size_t threads_tot = N;
  const std::size_t nblocks      = ( threads_tot + blocksize - 1 ) / blocksize;

  T *host_array          = (T *) malloc( N * sizeof( T ) );
  T *host_array_positive = (T *) malloc( N * sizeof( T ) );
 
  //host_array_initialize ( host_array, N );

  T epsilon = 1e-6;
  
  srand(time(0));
  for(std::size_t i = 0; i < N; i++){
    host_array[i] = common::initialize_random ( epsilon );
  } 

  std::size_t host_count = collect_positive_serial_host ( host_array, host_array_positive, N );

  const std::size_t m_default_device = omp_get_default_device();
  const std::size_t m_initial_device = omp_get_initial_device();
  const std::size_t m_offset = 0;
 
  T*        devc_array{nullptr};
  T*        devc_array_positive{nullptr};
  std::size_t* devc_count{nullptr};
  devc_array          = (T *) omp_target_alloc( N * sizeof( T ), m_default_device);
  devc_array_positive = (T *) omp_target_alloc( N * sizeof( T ), m_default_device);
  devc_count          = (std::size_t *) omp_target_alloc( sizeof( std::size_t ), m_default_device);

  if ( NULL == devc_array or NULL == devc_array_positive ) {
    std::cout << " ERROR: No space left on device." << std::endl;
  }

  if ( omp_target_memcpy( devc_array, host_array, N * sizeof( T ), 
			  m_offset, m_offset, m_default_device, m_initial_device ) ) {
       std::cout << "ERROR: copy random numbers from cpu to gpu " << std::endl;
  }

  //collect_positive_devc ( devc_array, devc_array_positive, devc_count, N, nblocks, blocksize );
  //BENCHMARK("OpenMP Atomic Capture") { return collect_positive_devc ( devc_array, devc_array_positive, devc_count, N, nblocks, blocksize ); };

  BENCHMARK_ADVANCED("OpenMP Atomic Capture")(Catch::Benchmark::Chronometer meter) {
    #pragma omp target is_device_ptr ( devc_count )
      devc_count[0] = 0;
    #pragma omp target is_device_ptr ( devc_array_positive ) 
    #pragma omp teams distribute parallel for 
    for ( std::size_t i = 0; i < N; i++ ) {
  	devc_array_positive[i] = 0; 
    }
    meter.measure([devc_array, devc_array_positive, devc_count, N, nblocks, blocksize] 
    { return collect_positive_devc ( devc_array, devc_array_positive, devc_count, N, nblocks, blocksize ); });
  };
  
  std::size_t host_copy_count = 0;
  if ( omp_target_memcpy( &host_copy_count, devc_count, sizeof( std::size_t ),
                          m_offset, m_offset, m_initial_device, m_default_device ) ) {
       std::cout << "ERROR: copy count from gpu to cpu " << std::endl;
  }

  T *host_copy_array = (T *) malloc( N * sizeof( T ) );
  if ( omp_target_memcpy( host_copy_array, devc_array_positive, host_count * sizeof( T ),
        		  m_offset, m_offset, m_initial_device, m_default_device ) ) {
       std::cout << "ERROR: copy random numbers from gpu to cpu " << std::endl;
  }

  CHECK( host_copy_count == host_count );

  /* * * check GPU array parallel with CPU array serial * * */
  T sum = 0.;
  for ( std::size_t i = 0; i < host_count; i++ ) {
    //std::cout << i << " " << host_copy_array[i] << " " << host_array_positive[i] << std::endl;  
    sum += host_copy_array[i] - host_array_positive[i];
  }

  CHECK( std::fabs ( (float)sum ) < 0.0001f );
 
  free( host_copy_array     );
  free( host_array          );
  free( host_array_positive );
  omp_target_free( devc_array,          m_default_device );
  omp_target_free( devc_array_positive, m_default_device );
  omp_target_free( devc_count,          m_default_device );

  return host_array;
}


} //namespace openmp_bench
