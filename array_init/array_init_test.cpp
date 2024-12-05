#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif

#include <openmp_bench.h>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

namespace openmp_bench {

/*******************/	
/* Array size 2^12 */
/*******************/	
#define N 1024*4

#define blocksize 128
TEST_CASE("OpenMP ArrayInit 2^12 128 double ") { double* temp1 = set_to_zero_wrapper <double> ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^12 128 float  ") { float*  temp2 = set_to_zero_wrapper <float>  ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^12 128 int    ") { int*    temp3 = set_to_zero_wrapper <int>    ( N, blocksize ); }
#undef blocksize

#define blocksize 256
TEST_CASE("OpenMP ArrayInit 2^12 256 double ") { double* temp1 = set_to_zero_wrapper <double> ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^12 256 float  ") { float*  temp2 = set_to_zero_wrapper <float>  ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^12 256 int    ") { int*    temp3 = set_to_zero_wrapper <int>    ( N, blocksize ); }
#undef blocksize

#define blocksize 512
TEST_CASE("OpenMP ArrayInit 2^12 512 double ") { double* temp1 = set_to_zero_wrapper <double> ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^12 512 float  ") { float*  temp2 = set_to_zero_wrapper <float>  ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^12 512 int    ") { int*    temp3 = set_to_zero_wrapper <int>    ( N, blocksize ); }
#undef blocksize

#define blocksize 1024
TEST_CASE("OpenMP ArrayInit 2^12 1024 double ") { double* temp1 = set_to_zero_wrapper <double> ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^12 1024 float  ") { float*  temp2 = set_to_zero_wrapper <float>  ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^12 1024 int    ") { int*    temp3 = set_to_zero_wrapper <int>    ( N, blocksize ); }
#undef blocksize

#undef N

/*******************/	
/* Array size 2^18 */
/*******************/	
#define N 1024*512

#define blocksize 128
TEST_CASE("OpenMP ArrayInit 2^18 128 double ") { double* temp1 = set_to_zero_wrapper <double> ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^18 128 float  ") { float*  temp2 = set_to_zero_wrapper <float>  ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^18 128 int    ") { int*    temp3 = set_to_zero_wrapper <int>    ( N, blocksize ); }
#undef blocksize

#define blocksize 256
TEST_CASE("OpenMP ArrayInit 2^18 256 double ") { double* temp1 = set_to_zero_wrapper <double> ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^18 256 float  ") { float*  temp2 = set_to_zero_wrapper <float>  ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^18 256 int    ") { int*    temp3 = set_to_zero_wrapper <int>    ( N, blocksize ); }
#undef blocksize

#define blocksize 512
TEST_CASE("OpenMP ArrayInit 2^18 512 double ") { double* temp1 = set_to_zero_wrapper <double> ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^18 512 float  ") { float*  temp2 = set_to_zero_wrapper <float>  ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^18 512 int    ") { int*    temp3 = set_to_zero_wrapper <int>    ( N, blocksize ); }
#undef blocksize

#define blocksize 1024
TEST_CASE("OpenMP ArrayInit 2^18 1024 double ") { double* temp1 = set_to_zero_wrapper <double> ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^18 1024 float  ") { float*  temp2 = set_to_zero_wrapper <float>  ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^18 1024 int    ") { int*    temp3 = set_to_zero_wrapper <int>    ( N, blocksize ); }
#undef blocksize

#undef N

/*******************/	
/* Array size 2^24 */
/*******************/	
#define N 1024*1024*16

#define blocksize 128
TEST_CASE("OpenMP ArrayInit 2^24 128 double ") { double* temp1 = set_to_zero_wrapper <double> ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^24 128 float  ") { float*  temp2 = set_to_zero_wrapper <float>  ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^24 128 int    ") { int*    temp3 = set_to_zero_wrapper <int>    ( N, blocksize ); }
#undef blocksize

#define blocksize 256
TEST_CASE("OpenMP ArrayInit 2^24 256 double ") { double* temp1 = set_to_zero_wrapper <double> ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^24 256 float  ") { float*  temp2 = set_to_zero_wrapper <float>  ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^24 256 int    ") { int*    temp3 = set_to_zero_wrapper <int>    ( N, blocksize ); }
#undef blocksize

#define blocksize 512
TEST_CASE("OpenMP ArrayInit 2^24 512 double ") { double* temp1 = set_to_zero_wrapper <double> ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^24 512 float  ") { float*  temp2 = set_to_zero_wrapper <float>  ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^24 512 int    ") { int*    temp3 = set_to_zero_wrapper <int>    ( N, blocksize ); }
#undef blocksize

#define blocksize 1024
TEST_CASE("OpenMP ArrayInit 2^24 1024 double ") { double* temp1 = set_to_zero_wrapper <double> ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^24 1024 float  ") { float*  temp2 = set_to_zero_wrapper <float>  ( N, blocksize ); }
TEST_CASE("OpenMP ArrayInit 2^24 1024 int    ") { int*    temp3 = set_to_zero_wrapper <int>    ( N, blocksize ); }
#undef blocksize

#undef N

} // namespace openmp_bench
