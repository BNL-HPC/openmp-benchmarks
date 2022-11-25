#include <catch.hpp> 
#include <cuda_bench.cuh>
#include <iostream>
#include <cuda.h>

namespace cuda_bench {

TEST_CASE("CUDA Test"){

  double* temp1 = set_to_zero_wrapper <double> ( 4096, 512); // N, blocksize 
  float*  temp2 = set_to_zero_wrapper <float>  ( 4096, 512); // N, blocksize 
  int*    temp3 = set_to_zero_wrapper <int>    ( 4096, 512); // N, blocksize 
}

} // namespace cuda_bench
