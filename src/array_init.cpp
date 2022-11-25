#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif

#include <openmp_bench.h>
#include <catch.hpp> 
#include <iostream>
#include <omp.h>

namespace openmp_bench {

TEST_CASE("OpenMP Test") {
   
  double* temp1 = set_to_zero_wrapper <double> (4096, 256);
  float*  temp2 = set_to_zero_wrapper <float>  (4096, 256);
  int*    temp3 = set_to_zero_wrapper <int>    (4096, 256);
  	
}

} // namespace openmp_bench
