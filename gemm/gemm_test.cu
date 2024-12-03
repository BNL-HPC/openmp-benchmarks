#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bench.cuh>
#include <catch2/catch_test_macros.hpp>

namespace cuda_bench {

/*******************/	
/* Array size 2^10 */
/*******************/	
#define N 1024

#define blocksize 32
TEST_CASE("CUDA GEMM 1024x1024x1024 32 double"){ double* temp1 = gemm_wrapper <double> (N, N, N, blocksize); }
TEST_CASE("CUDA GEMM 1024x1024x1024 32 float" ){ float*  temp1 = gemm_wrapper <float>  (N, N, N, blocksize); }
//TEST_CASE("CUDA GEMM 8192x8192x8192 double"){ double* temp1 = gemm_wrapper <double> (N, N, N); }
//TEST_CASE("CUDA GEMM 8192x8192x8192 float" ){ float*  temp1 = gemm_wrapper <float>  (N, N, N); }
#undef blocksize

} // namespace cuda_bench
