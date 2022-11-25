#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif

#include <openmp_bench.h>
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

TEST_CASE("my test") {

  const int N = 4096;	
  openmp_bench <double, N, 256> x;

  /* Init device array with zero */
  BENCHMARK("OpenMP Array Init") { return x.set_array_zero ( ); };
  
  /* Copy device array to host for tests */
  double *host_array = (double *) malloc( N * sizeof( double ) );
  if ( omp_target_memcpy( host_array, x.m_device_array, N * sizeof( double ),
                                  x.m_offset, x.m_offset, x.m_initial_device, x.m_default_device ) ) {
    std::cout << "ERROR: array " << std::endl;
  }

  for(int i = 0; i < N; i++)
    if ( std::fabs( host_array[i] ) > 1e-20 )
      std::cout << "!!Problem at i = " << i << std::endl;

  REQUIRE(14==14);
}

