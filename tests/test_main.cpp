#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch.hpp> 
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <chrono>
#include <array>
#include <vector>
#include <cmath>
#include <omp.h>

template <typename T>
inline void set_to_zero( T* device_array, const int N ) {

  int threads_tot = N;
  int blocksize   = 512;//704;
  int nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;

  /* Time target offload and init */
  auto start = std::chrono::steady_clock::now();

  int i;
  #pragma omp target is_device_ptr ( device_array )            
  #pragma omp teams distribute parallel for num_teams(nblocks) num_threads(blocksize) 
  for(i = 0; i < N; i++) {
    //printf(" num teams = %d, num threads = %d", omp_get_num_teams(), omp_get_num_threads() );
    device_array[i] = 0.;
  }

  std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - start;
  //std::cout << "time to init array = " << elapsed_seconds.count() << " sec" << std::endl;

  return;
}    

TEST_CASE("my test") {

  using real = double;
  srand(time(0));
  std::cout << "OpenMP num devices = " << omp_get_num_devices() << std::endl;

  const int m_default_device = omp_get_default_device();
  const int m_initial_device = omp_get_initial_device();
  const std::size_t m_offset = 0;
  const int N = 4096*64;

  /* Allocate array of length N on target */
  real *device_array = (real *) omp_target_alloc( N * sizeof( real ), m_default_device);
  if ( device_array == NULL ) {
    std::cout << " ERROR: No space left on device." << std::endl;
  }

  /* Init device array with zero */
  BENCHMARK("OpenMP Array Init") { return set_to_zero ( device_array, N ); };

  /* Copy device array to host for tests */
  real *host_array = (real *) malloc( N * sizeof( real ) );
  if ( omp_target_memcpy( host_array, device_array, N * sizeof( real ),
                                  m_offset, m_offset, m_initial_device, m_default_device ) ) {
    std::cout << "ERROR: array " << std::endl;
  }

  for(int i = 0; i < N; i++)
    if ( std::fabs( host_array[i] ) > 1e-20 )
      std::cout << "!!Problem at i = " << i << std::endl;




  REQUIRE(14==14);
}

