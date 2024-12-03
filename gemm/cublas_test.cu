#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bench.cuh>
#include <catch2/catch_test_macros.hpp>

namespace cuda_bench {

/*******************/	
/* Array size 2^8  */
/*******************/	
#define N 256

TEST_CASE("cuBLAS GEMM 256x256x256 double"){ double* temp1 = cublas_wrapper_d <double> (N, N, N); }
TEST_CASE("cuBLAS GEMM 256x256x256 float" ){ float*  temp1 = cublas_wrapper_f <float>  (N, N, N); }

#undef N

/*******************/	
/* Array size 2^9  */
/*******************/	
#define N 512

TEST_CASE("cuBLAS GEMM 512x512x512 double"){ double* temp1 = cublas_wrapper_d <double> (N, N, N); }
TEST_CASE("cuBLAS GEMM 512x512x512 float" ){ float*  temp1 = cublas_wrapper_f <float>  (N, N, N); }

#undef N

/*******************/	
/* Array size 2^10 */
/*******************/	
#define N 1024

TEST_CASE("cuBLAS GEMM 1024x1024x1024 double"){ double* temp1 = cublas_wrapper_d <double> (N, N, N); }
TEST_CASE("cuBLAS GEMM 1024x1024x1024 float" ){ float*  temp1 = cublas_wrapper_f <float>  (N, N, N); }

#undef N

/*******************/	
/* Array size 2^11 */
/*******************/	
#define N 2048

TEST_CASE("cuBLAS GEMM 2048x2048x2048 double"){ double* temp1 = cublas_wrapper_d <double> (N, N, N); }
TEST_CASE("cuBLAS GEMM 2048x2048x2048 float" ){ float*  temp1 = cublas_wrapper_f <float>  (N, N, N); }

#undef N

/*******************/	
/* Array size 2^12  */
/*******************/	
#define N 4096

TEST_CASE("cuBLAS GEMM 4096x4096x4096 double"){ double* temp1 = cublas_wrapper_d <double> (N, N, N); }
TEST_CASE("cuBLAS GEMM 4096x4096x4096 float" ){ float*  temp1 = cublas_wrapper_f <float>  (N, N, N); }

#undef N







/*******************/	
/* Array size 2^13 */
/*******************/	
#define N 8192

TEST_CASE("cuBLAS GEMM 8192x8192x8192 double"){ double* temp1 = cublas_wrapper_d <double> (N, N, N); }
TEST_CASE("cuBLAS GEMM 8192x8192x8192 float" ){ float*  temp1 = cublas_wrapper_f <float>  (N, N, N); }

#undef N

} // namespace cuda_bench
