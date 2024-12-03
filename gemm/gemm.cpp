#define CATCH_CONFIG_ENABLE_BENCHMARKING
#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#endif

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <openmp_bench.h>
#include <common.hpp>
#include <omp.h>

namespace openmp_bench {

template double* gemm_wrapper <double> ( const std::size_t, const std::size_t, const std::size_t, const std::size_t);
template float*  gemm_wrapper <float>  ( const std::size_t, const std::size_t, const std::size_t, const std::size_t);
//template int*    gemm_wrapper <int>    ( const std::size_t, const std::size_t, const std::size_t);

template <typename T>
void gemm_naive (T* d_A, T* d_B, T* d_C, const std::size_t M, const std::size_t N, const std::size_t K, const T alpha, const T beta, const std::size_t nblocks, const std::size_t blocksize) {
  #pragma omp target is_device_ptr ( d_A, d_B, d_C ) 
  #pragma omp teams distribute parallel for num_teams(nblocks) num_threads(blocksize) 
  for ( std::size_t i = 0; i < M; i++ ) 
  for ( std::size_t k = 0; k < K; k++ )  
  for ( std::size_t j = 0; j < N; j++ ) {
    d_C[i*N + j] = alpha * d_A[i*K + k] * d_B[k*N + j] + beta * d_C[i*N + j]; 
  }
}

template <typename T>
T* gemm_wrapper ( const std::size_t M, const std::size_t N, const std::size_t K, const std::size_t blocksize ) {
  //https://gist.github.com/nadavrot/5b35d44e8ba3dd718e595e40184d03f0  
  const std::size_t threads_tot = M; // parallelize outermost loop
  // TODO each element of the result matrix can be computed by an independent thread
  const std::size_t nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;

  check_target_device ();
  
  T* h_A = (T*)malloc(sizeof(T) * M * K);
  T* h_B = (T*)malloc(sizeof(T) * K * N);
  T* h_C = (T*)malloc(sizeof(T) * M * N);

  T epsilon = common::get_epsilon <T> ();
  srand(time(0));
  for (int i = 0; i < M*K; i++)
    h_A[i] = common::initialize_random ( epsilon );

  for (int i = 0; i < K*N; i++)
    h_B[i] = common::initialize_random ( epsilon );
 
  for (int i = 0; i < M*N; i++)
    h_C[i] = common::initialize_random ( epsilon );
 
  const std::size_t m_default_device = omp_get_default_device();
  const std::size_t m_initial_device = omp_get_initial_device();
  const std::size_t m_offset = 0;

  BENCHMARK_ADVANCED("CUDA GEMM")(Catch::Benchmark::Chronometer meter) {

    T* d_A{nullptr};
    T* d_B{nullptr};
    T* d_C{nullptr};
    d_A = (T *) omp_target_alloc( M * K * sizeof( T ), m_default_device);
    d_B = (T *) omp_target_alloc( K * N * sizeof( T ), m_default_device);
    d_C = (T *) omp_target_alloc( M * N * sizeof( T ), m_default_device);

    if ( NULL == d_A or NULL == d_B or NULL == d_C ) 
      std::cout << " ERROR: No space left on device." << std::endl;

    if ( omp_target_memcpy( d_A, h_A, M * K * sizeof( T ), 
          		  m_offset, m_offset, m_default_device, m_initial_device ) ) {
         std::cout << "ERROR: copy random numbers from cpu to gpu " << std::endl;
    }
    if ( omp_target_memcpy( d_B, h_B, K * N * sizeof( T ), 
          		  m_offset, m_offset, m_default_device, m_initial_device ) ) {
         std::cout << "ERROR: copy random numbers from cpu to gpu " << std::endl;
    }
    if ( omp_target_memcpy( d_C, h_C, M * N * sizeof( T ), 
          		  m_offset, m_offset, m_default_device, m_initial_device ) ) {
         std::cout << "ERROR: copy random numbers from cpu to gpu " << std::endl;
    }

    #pragma omp target is_device_ptr ( d_C ) 
    for ( std::size_t i = 0; i < M*N; i++ ) 
      d_C[i] = 0;

    T alpha(1.0), beta(0.5);

    double start, secTotal;
    start = omp_get_wtime();
    meter.measure( [N, M, K, &alpha, &beta, d_B, d_A, d_C, nblocks, blocksize] { 
      gemm_naive(d_B, d_A, d_C, M, N, K, alpha, beta, nblocks, blocksize);
    });
    secTotal = omp_get_wtime() - start;

    printf ("OpenMP performance : time = %7.3f ns, GFLOPs/sec = %.4f \n", secTotal*1e9, 
          	  2*M*N*K*1.0e-9/(secTotal) );

    if ( omp_target_memcpy( h_C, d_C, M * N * sizeof( T ),
          		  m_offset, m_offset, m_initial_device, m_default_device ) ) {
         std::cout << "ERROR: copy C from gpu to cpu " << std::endl;
    }

    omp_target_free( d_A, m_default_device );
    omp_target_free( d_B, m_default_device );
    omp_target_free( d_C, m_default_device );
  };

  // print matrix
  //auto print_matrix = [] (T* mat, int rows, int cols) {
  //  for (int i = 0; i < rows; i++) {
  //    for (int j = 0; j < cols; j++) {
  //      std::cout << mat[i*cols + j] << " " ;
  //    }
  //    std::cout << std::endl;
  //  }
  //};

  //std::cout << "A matrix is \n";
  //print_matrix (h_A, M, K) ;
  //std::cout << "B matrix is \n";
  //print_matrix (h_B, K, N) ;
  //std::cout << "C matrix is \n";
  //print_matrix (h_C, M, N) ;
	 
  free(h_A);
  free(h_B);
  free(h_C);
  
  return h_A; 
}

} //namespace openmp_bench
