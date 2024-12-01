#define CATCH_CONFIG_ENABLE_BENCHMARKING
#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#endif

#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bench.cuh>
#include <catch.hpp>
#include <common.hpp>

namespace cuda_bench {

template double* gemm_wrapper <double> ( const int, const int, const int);
//template float*  sgemm_wrapper <float>  ( const int, const int, const int);
//template int*    sgemm_wrapper <int>    ( const int, const int, const int);

template<typename T>
__global__ void gemm_naive(int M, int N, int K, T alpha, const T *A,
                            const T *B, T beta, T *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    T tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = tmp;
    //C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

template<typename T, int BLOCKSIZE>
__global__ void gemm_global_mem_coalesce(int M, int N, int K, T alpha,
                                          const T *A, const T *B,
                                          T beta, T *C) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // if statement is necessary to make things work under tile quantization
  if (cRow < M && cCol < N) {
    T tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = tmp;
    //C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}

// matrix multiplication C = alpha * A * B + beta * C
template<typename T>
__host__ T* gemm_wrapper ( const int M, const int N, const int K ) {
  cudaError_t cudaStat;  // cudaMalloc status

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

  cudaMemcpy(d_A, h_A, M * K * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(T), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_c, c, m * n * sizeof(float), cudaMemcpyHostToDevice);  

  T alpha(1.0), beta(0.0);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  dim3 gridDim(M/32+1, N/32+1);
  dim3 blockDim(32, 32);
  gemm_naive<<<gridDim, blockDim >>> (M, N, K, alpha, d_A, d_B, beta, d_C);
  //gemm_global_mem_coalesce <T, 32> <<<gridDim, blockDim >>> (M, N, K, alpha, d_A, d_B, beta, d_C);
  //stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N,
  //                   d_A, K, &beta, d_C, N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float msecTotal(0.0f);
  cudaEventElapsedTime(&msecTotal, start, stop);

  printf ("CUDA Naive performance : time = %.3f, GFLOPs/sec = %.4f ", msecTotal, 
		  2*M*N*K*1.0e-9/(msecTotal/1000.0) );

  cudaMemcpy(h_C, d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return h_A; 
}



} // namespace cuda_bench