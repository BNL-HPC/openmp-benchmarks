#include <catch.hpp> 
#include <hip/hip_runtime.h>
#include <hip_bench.h>
#include <iostream>

namespace hip_bench {

/*******************/	
/* Array size 2^12 */
/*******************/	

TEST_CASE("HIP Init Array to 0 2^12 128"){

  double* temp1 = set_to_zero_wrapper <double> ( 4096, 128); // N, blocksize 
  float*  temp2 = set_to_zero_wrapper <float>  ( 4096, 128); // N, blocksize 
  int*    temp3 = set_to_zero_wrapper <int>    ( 4096, 128); // N, blocksize 
}

TEST_CASE("HIP Init Array to 0 2^12 256"){

  double* temp1 = set_to_zero_wrapper <double> ( 4096, 256); // N, blocksize 
  float*  temp2 = set_to_zero_wrapper <float>  ( 4096, 256); // N, blocksize 
  int*    temp3 = set_to_zero_wrapper <int>    ( 4096, 256); // N, blocksize 
}

TEST_CASE("HIP Init Array to 0 2^12 512"){

  double* temp1 = set_to_zero_wrapper <double> ( 4096, 512); // N, blocksize 
  float*  temp2 = set_to_zero_wrapper <float>  ( 4096, 512); // N, blocksize 
  int*    temp3 = set_to_zero_wrapper <int>    ( 4096, 512); // N, blocksize 
}

TEST_CASE("HIP Init Array to 0 2^12 1024"){

  double* temp1 = set_to_zero_wrapper <double> ( 4096, 1024); // N, blocksize 
  float*  temp2 = set_to_zero_wrapper <float>  ( 4096, 1024); // N, blocksize 
  int*    temp3 = set_to_zero_wrapper <int>    ( 4096, 1024); // N, blocksize 
}


/*******************/	
/* Array size 2^18 */
/*******************/	

TEST_CASE("HIP Init Array to 0 2^18 128"){

  double* temp1 = set_to_zero_wrapper <double> ( 4096*64, 128); // N, blocksize 
  float*  temp2 = set_to_zero_wrapper <float>  ( 4096*64, 128); // N, blocksize 
  int*    temp3 = set_to_zero_wrapper <int>    ( 4096*64, 128); // N, blocksize 
}

TEST_CASE("HIP Init Array to 0 2^18 256"){

  double* temp1 = set_to_zero_wrapper <double> ( 4096*64, 256); // N, blocksize 
  float*  temp2 = set_to_zero_wrapper <float>  ( 4096*64, 256); // N, blocksize 
  int*    temp3 = set_to_zero_wrapper <int>    ( 4096*64, 256); // N, blocksize 
}

TEST_CASE("HIP Init Array to 0 2^18 512"){

  double* temp1 = set_to_zero_wrapper <double> ( 4096*64, 512); // N, blocksize 
  float*  temp2 = set_to_zero_wrapper <float>  ( 4096*64, 512); // N, blocksize 
  int*    temp3 = set_to_zero_wrapper <int>    ( 4096*64, 512); // N, blocksize 
}

TEST_CASE("HIP Init Array to 0 2^18 1024"){

  double* temp1 = set_to_zero_wrapper <double> ( 4096*64, 1024); // N, blocksize 
  float*  temp2 = set_to_zero_wrapper <float>  ( 4096*64, 1024); // N, blocksize 
  int*    temp3 = set_to_zero_wrapper <int>    ( 4096*64, 1024); // N, blocksize 
}


/*******************/	
/* Array size 2^24 */
/*******************/	

TEST_CASE("HIP Init Array to 0 2^24 128"){

  double* temp1 = set_to_zero_wrapper <double> ( 4096*4096, 128); // N, blocksize 
  float*  temp2 = set_to_zero_wrapper <float>  ( 4096*4096, 128); // N, blocksize 
  int*    temp3 = set_to_zero_wrapper <int>    ( 4096*4096, 128); // N, blocksize 
}

TEST_CASE("HIP Init Array to 0 2^24 256"){

  double* temp1 = set_to_zero_wrapper <double> ( 4096*4096, 256); // N, blocksize 
  float*  temp2 = set_to_zero_wrapper <float>  ( 4096*4096, 256); // N, blocksize 
  int*    temp3 = set_to_zero_wrapper <int>    ( 4096*4096, 256); // N, blocksize 
}

TEST_CASE("HIP Init Array to 0 2^24 512"){

  double* temp1 = set_to_zero_wrapper <double> ( 4096*4096, 512); // N, blocksize 
  float*  temp2 = set_to_zero_wrapper <float>  ( 4096*4096, 512); // N, blocksize 
  int*    temp3 = set_to_zero_wrapper <int>    ( 4096*4096, 512); // N, blocksize 
}

TEST_CASE("HIP Init Array to 0 2^24 1024"){

  double* temp1 = set_to_zero_wrapper <double> ( 4096*4096, 1024); // N, blocksize 
  float*  temp2 = set_to_zero_wrapper <float>  ( 4096*4096, 1024); // N, blocksize 
  int*    temp3 = set_to_zero_wrapper <int>    ( 4096*4096, 1024); // N, blocksize 
}



} // namespace hip_bench
