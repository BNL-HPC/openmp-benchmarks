#include <openmp_bench.h>
#include <catch.hpp>
#include <omp.h>

namespace openmp_bench {

/*******************/
/* Array size 2^12 */
/*******************/

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

TEST_CASE("OpenMP Atomic Capture, 2^12, 1024"){

  const std::size_t N = 4096;
  const std::size_t blocksize = 1024;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );
}

/*******************/	
/* Array size 2^16 */
/*******************/	

TEST_CASE("OpenMP Atomic Capture, 2^16, 128"){

  const std::size_t N = 4096*16; 
  const std::size_t blocksize = 128;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("OpenMP Atomic Capture, 2^16, 256"){

  const std::size_t N = 4096*16; 
  const std::size_t blocksize = 256;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("OpenMP Atomic Capture, 2^16, 512"){

  const std::size_t N = 4096*16; 
  const std::size_t blocksize = 512;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("OpenMP Atomic Capture, 2^16, 1024"){

  const std::size_t N = 4096*16; 
  const std::size_t blocksize = 1024;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

/*******************/	
/* Array size 2^20 */
/*******************/	

TEST_CASE("OpenMP Atomic Capture, 2^20, 128"){

  const std::size_t N = 4096*256; 
  const std::size_t blocksize = 128;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("OpenMP Atomic Capture, 2^20, 256"){

  const std::size_t N = 4096*256; 
  const std::size_t blocksize = 256;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("OpenMP Atomic Capture, 2^20, 512"){

  const std::size_t N = 4096*256; 
  const std::size_t blocksize = 512;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("OpenMP Atomic Capture, 2^20, 1024"){

  const std::size_t N = 4096*256; 
  const std::size_t blocksize = 1024;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

/*******************/	
/* Array size 2^24 */
/*******************/	

TEST_CASE("OpenMP Atomic Capture, 2^24, 128"){

  const std::size_t N = 4096*4096; 
  const std::size_t blocksize = 128;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("OpenMP Atomic Capture, 2^24, 256"){

  const std::size_t N = 4096*4096; 
  const std::size_t blocksize = 256;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("OpenMP Atomic Capture, 2^24, 512"){

  const std::size_t N = 4096*4096; 
  const std::size_t blocksize = 512;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("OpenMP Atomic Capture, 2^24, 1024"){

  const std::size_t N = 4096*4096; 
  const std::size_t blocksize = 1024;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

/*******************/	
/* Array size 2^28 */
/*******************/	

TEST_CASE("OpenMP Atomic Capture, 2^28, 128"){

  const std::size_t N = 4096*4096*16; 
  const std::size_t blocksize = 128;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("OpenMP Atomic Capture, 2^28, 256"){

  const std::size_t N = 4096*4096*16; 
  const std::size_t blocksize = 256;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("OpenMP Atomic Capture, 2^28, 512"){

  const std::size_t N = 4096*4096*16; 
  const std::size_t blocksize = 512;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}

TEST_CASE("OpenMP Atomic Capture, 2^28, 1024"){

  const std::size_t N = 4096*4096*16; 
  const std::size_t blocksize = 1024;
  double* temp1 = atomic_capture_wrapper <double> ( N, blocksize );  
  float*  temp2 = atomic_capture_wrapper <float>  ( N, blocksize );  
  int*    temp3 = atomic_capture_wrapper <int>    ( N, blocksize );  
}





} //namespace openmp_bench
