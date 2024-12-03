#include <openmp_bench.h>
#include <catch2/catch_test_macros.hpp>

namespace openmp_bench {
/*******************/	
/* Array size 2^10 */
/*******************/	
#define N 1024

#define blocksize 32
TEST_CASE("OpenMP GEMM 1024x1024x1024 32 double"){ double* temp1 = gemm_wrapper <double> (N, N, N, blocksize); }
TEST_CASE("OpenMP GEMM 1024x1024x1024 32 float" ){ float*  temp1 = gemm_wrapper <float>  (N, N, N, blocksize); }
#undef blocksize

} // namespace 
