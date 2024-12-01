#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bench.cuh>
#include <catch.hpp>

namespace cuda_bench {

/*******************/	
/* Array size 2^12 */
/*******************/	

TEST_CASE("CUDA GEMM 4x4x4"){

  const int M = 512; 
  const int N = 512; 
  const int K = 512; 
  double* temp1 = gemm_wrapper <double> (M, N, K);  
  //float*  temp2 = reduction_wrapper <float>  ( M, N, K );  
  //int*    temp3 = reduction_wrapper <int>    ( N, blocksize );  
}

} // namespace cuda_bench
