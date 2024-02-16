#define CATCH_CONFIG_ENABLE_BENCHMARKING
#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#endif

#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bench.cuh>
#include <catch.hpp>
#include <common.hpp>

namespace cuda_bench {

template double* gemm_wrapper <double> ( const int, const int, const int);
//template float*  sgemm_wrapper <float>  ( const int, const int, const int);
//template int*    sgemm_wrapper <int>    ( const int, const int, const int);



// matrix multiplication C = alpha * A * B + beta * C
template<typename T>
__host__ T* gemm_wrapper ( const int M, const int N, const int K ) {

  cudaError_t cudaStat;  // cudaMalloc status
  cublasStatus_t stat;   // cuBLAS functions status
  cublasHandle_t handle; // cuBLAS context

  T* h_A = (T*)malloc(sizeof(T) * M * K);
  T* h_B = (T*)malloc(sizeof(T) * K * N);
  T* h_C = (T*)malloc(sizeof(T) * M * N);

  T epsilon = common::get_epsilon <T> ();
  srand(time(0));
  for (int i = 0; i < M*K; i++)
    h_A[i] = common::initialize_random ( epsilon );

  for (int i = 0; i < K*N; i++)
    h_B[i] = common::initialize_random ( epsilon );

  T *d_A, *d_B, *d_C;
  cudaMalloc( (void**) &d_A, sizeof(T) * M * K);
  cudaMalloc( (void**) &d_B, sizeof(T) * K * N);
  cudaMalloc( (void**) &d_C, sizeof(T) * M * N);

  stat = cublasCreate(&handle); // initialize CUBLAS context

  cudaMemcpy(d_A, h_A, M * K * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(T), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_c, c, m * n * sizeof(float), cudaMemcpyHostToDevice);  

  T alpha(1.0), beta(0.0);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N,
                     d_A, K, &beta, d_C, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float msecTotal(0.0f);
  cudaEventElapsedTime(&msecTotal, start, stop);

  printf ("cuBLAS performance : time = %.3f, GFLOPs/sec = %.4f ", msecTotal, 
		  2*M*N*K*1.0e-9/(msecTotal/1000.0) );

  cudaMemcpy(h_C, d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(handle); // destroy CUBLAS context
  free(h_A);
  free(h_B);
  free(h_C);

  return h_A; 
}



} // namespace cuda_bench
