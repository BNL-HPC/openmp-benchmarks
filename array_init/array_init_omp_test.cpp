#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif

#include <openmp_bench.h>
#include <catch.hpp> 
#include <iostream>
#include <omp.h>

namespace openmp_bench {

/*******************/	
/* Array size 2^12 */
/*******************/	

TEST_CASE("OpenMP Array Init to 0, 2^12, 128") {
   
  double* temp1 = set_to_zero_wrapper <double> (4096, 128);
  float*  temp2 = set_to_zero_wrapper <float>  (4096, 128);
  int*    temp3 = set_to_zero_wrapper <int>    (4096, 128);
}

TEST_CASE("OpenMP Array Init to 0, 2^12, 256") {
   
  double* temp1 = set_to_zero_wrapper <double> (4096, 256);
  float*  temp2 = set_to_zero_wrapper <float>  (4096, 256);
  int*    temp3 = set_to_zero_wrapper <int>    (4096, 256);
}

TEST_CASE("OpenMP Array Init to 0, 2^12, 512") {
   
  double* temp1 = set_to_zero_wrapper <double> (4096, 512);
  float*  temp2 = set_to_zero_wrapper <float>  (4096, 512);
  int*    temp3 = set_to_zero_wrapper <int>    (4096, 512);
}

TEST_CASE("OpenMP Array Init to 0, 2^12, 768") {
   
  double* temp1 = set_to_zero_wrapper <double> (4096, 768);
  float*  temp2 = set_to_zero_wrapper <float>  (4096, 768);
  int*    temp3 = set_to_zero_wrapper <int>    (4096, 768);
}


/*******************/	
/* Array size 2^18 */
/*******************/	

TEST_CASE("OpenMP Array Init to 0, 2^18, 128") {
   
  double* temp1 = set_to_zero_wrapper <double> (4096*64, 128);
  float*  temp2 = set_to_zero_wrapper <float>  (4096*64, 128);
  int*    temp3 = set_to_zero_wrapper <int>    (4096*64, 128);
}

TEST_CASE("OpenMP Array Init to 0, 2^18, 256") {
   
  double* temp1 = set_to_zero_wrapper <double> (4096*64, 256);
  float*  temp2 = set_to_zero_wrapper <float>  (4096*64, 256);
  int*    temp3 = set_to_zero_wrapper <int>    (4096*64, 256);
}

TEST_CASE("OpenMP Array Init to 0, 2^18, 512") {
   
  double* temp1 = set_to_zero_wrapper <double> (4096*64, 512);
  float*  temp2 = set_to_zero_wrapper <float>  (4096*64, 512);
  int*    temp3 = set_to_zero_wrapper <int>    (4096*64, 512);
}

TEST_CASE("OpenMP Array Init to 0, 2^18, 768") {
   
  double* temp1 = set_to_zero_wrapper <double> (4096*64, 768);
  float*  temp2 = set_to_zero_wrapper <float>  (4096*64, 768);
  int*    temp3 = set_to_zero_wrapper <int>    (4096*64, 768);
}

/*******************/	
/* Array size 2^24 */
/*******************/	

TEST_CASE("OpenMP Array Init to 0, 2^24, 128") {
   
  double* temp1 = set_to_zero_wrapper <double> (4096*4096, 128);
  float*  temp2 = set_to_zero_wrapper <float>  (4096*4096, 128);
  int*    temp3 = set_to_zero_wrapper <int>    (4096*4096, 128);
}

TEST_CASE("OpenMP Array Init to 0, 2^24, 256") {
   
  double* temp1 = set_to_zero_wrapper <double> (4096*4096, 256);
  float*  temp2 = set_to_zero_wrapper <float>  (4096*4096, 256);
  int*    temp3 = set_to_zero_wrapper <int>    (4096*4096, 256);
}

TEST_CASE("OpenMP Array Init to 0, 2^24, 512") {
   
  double* temp1 = set_to_zero_wrapper <double> (4096*4096, 512);
  float*  temp2 = set_to_zero_wrapper <float>  (4096*4096, 512);
  int*    temp3 = set_to_zero_wrapper <int>    (4096*4096, 512);
}

TEST_CASE("OpenMP Array Init to 0, 2^24, 768") {
   
  double* temp1 = set_to_zero_wrapper <double> (4096*4096, 768);
  float*  temp2 = set_to_zero_wrapper <float>  (4096*4096, 768);
  int*    temp3 = set_to_zero_wrapper <int>    (4096*4096, 768);
}


} // namespace openmp_bench
