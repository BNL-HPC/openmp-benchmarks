#include <catch.hpp> 
#include <cuda_bench.cuh>
#include <iostream>
#include <cuda.h>

namespace cuda_bench {

TEST_CASE("CUDA test"){

  double* temp1 = test_wrapper <double> ( 4096, 512); // N, blocksize 
  float*  temp2 = test_wrapper <float>  ( 4096, 512); // N, blocksize 
  int*    temp3 = test_wrapper <int>    ( 4096, 512); // N, blocksize 
}

} // namespace cuda_bench
