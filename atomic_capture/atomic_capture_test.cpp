#include <openmp_bench.h>
#include <catch.hpp>
#include <omp.h>

namespace openmp_bench {

TEST_CASE("OpenMP Atomic Capture, 2^12, 128"){

  const std::size_t N = 4096; 
  const std::size_t blocksize = 128;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("OpenMP Atomic Capture, 2^12, 256"){

  const std::size_t N = 4096; 
  const std::size_t blocksize = 256;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("OpenMP Atomic Capture, 2^12, 512"){

  const std::size_t N = 4096; 
  const std::size_t blocksize = 512;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}


} //namespace openmp_bench
