#include <cuda.h>
#include <cuda_bench.cuh>
#include <catch.hpp>

namespace cuda_bench {

/*******************/	
/* Array size 2^12 */
/*******************/	

TEST_CASE("CUDA Atomic Capture, 2^12, 128"){

  const int N = 4096; 
  const int blocksize = 128;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("CUDA Atomic Capture, 2^12, 256"){

  const int N = 4096; 
  const int blocksize = 256;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("CUDA Atomic Capture, 2^12, 512"){

  const int N = 4096; 
  const int blocksize = 512;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("CUDA Atomic Capture, 2^12, 1024"){

  const int N = 4096; 
  const int blocksize = 1024;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

/*******************/	
/* Array size 2^18 */
/*******************/	

TEST_CASE("CUDA Atomic Capture, 2^18, 128"){

  const int N = 4096*64; 
  const int blocksize = 128;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("CUDA Atomic Capture, 2^18, 256"){

  const int N = 4096*64; 
  const int blocksize = 256;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("CUDA Atomic Capture, 2^18, 512"){

  const int N = 4096*64; 
  const int blocksize = 512;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("CUDA Atomic Capture, 2^18, 1024"){

  const int N = 4096*64; 
  const int blocksize = 1024;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

/*******************/	
/* Array size 2^24 */
/*******************/	

TEST_CASE("CUDA Atomic Capture, 2^24, 128"){

  const int N = 4096*4096; 
  const int blocksize = 128;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("CUDA Atomic Capture, 2^24, 256"){

  const int N = 4096*4096; 
  const int blocksize = 256;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("CUDA Atomic Capture, 2^24, 512"){

  const int N = 4096*4096; 
  const int blocksize = 512;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("CUDA Atomic Capture, 2^24, 1024"){

  const int N = 4096*4096; 
  const int blocksize = 1024;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}




} //namespace cuda_bench
