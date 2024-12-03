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
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <common.hpp>

namespace cuda_bench {

template double* cublas_wrapper_d <double> ( const int, const int, const int);
template float*  cublas_wrapper_f <float>  ( const int, const int, const int);
//template int*    sgemm_wrapper <int>    ( const int, const int, const int);


// matrix multiplication C = alpha * A * B + beta * C
template<typename T>
__host__ T* cublas_wrapper_f ( const int M, const int N, const int K ) {

  cudaError_t cudaStat;  // cudaMalloc status
  cublasStatus_t stat;   // cuBLAS functions status

  T* h_A = (T*)malloc(sizeof(T) * M * K);
  T* h_B = (T*)malloc(sizeof(T) * K * N);
  T* h_C = (T*)malloc(sizeof(T) * M * N);
  
  srand(time(0));
  T epsilon = common::get_epsilon <T> ();
  for (int i = 0; i < M*K; i++)
    h_A[i] = common::initialize_random ( epsilon );

  for (int i = 0; i < K*N; i++)
    h_B[i] = common::initialize_random ( epsilon );

  for (int i = 0; i < M*N; i++)
    h_C[i] = common::initialize_random ( epsilon );
 
  BENCHMARK_ADVANCED("cuBLAS SGEMM")(Catch::Benchmark::Chronometer meter) {

    T *d_A, *d_B, *d_C;
    cudaMalloc( (void**) &d_A, sizeof(T) * M * K);
    cudaMalloc( (void**) &d_B, sizeof(T) * K * N);
    cudaMalloc( (void**) &d_C, sizeof(T) * M * N);
    cublasHandle_t handle; // cuBLAS context

    stat = cublasCreate(&handle); // initialize CUBLAS context

    cudaMemcpy(d_A, h_A, M * K * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(T), cudaMemcpyHostToDevice);

    T alpha(1.0), beta(0.5);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    meter.measure( [handle, N, M, K, &alpha, &beta, d_B, d_A, d_C] { 
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N,
                     d_A, K, &beta, d_C, N); 
    cudaDeviceSynchronize() ;
    });
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msecTotal(0.0f);
    cudaEventElapsedTime(&msecTotal, start, stop);
    cudaMemcpy(h_C, d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost);

    double Gflops = 2.0 * M * N * K * 1e-9;
    double avg_time = msecTotal*1000000.0;
    //printf("Average elapsed time: (%7.6f) ns %f, %f, %f \n", avg_time, h_A[0], h_B[0], h_C[0]);
    //printf("Gflops = %f \n", Gflops);
    //printf("%f GFLOPs/s \n", Gflops/avg_time);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle); // destroy CUBLAS context
  };
  free(h_A);
  free(h_B);
  free(h_C);

  return h_A; 
}


template<typename T>
__host__ T* cublas_wrapper_d ( const int M, const int N, const int K ) {

  cudaError_t cudaStat;  // cudaMalloc status
  cublasStatus_t stat;   // cuBLAS functions status
  cublasHandle_t handle; // cuBLAS context

  T* h_A = (T*)malloc(sizeof(T) * M * K);
  T* h_B = (T*)malloc(sizeof(T) * K * N);
  T* h_C = (T*)malloc(sizeof(T) * M * N);
  
  srand(time(0));
  T epsilon = common::get_epsilon <T> ();
  for (int i = 0; i < M*K; i++)
    h_A[i] = common::initialize_random ( epsilon );

  for (int i = 0; i < K*N; i++)
    h_B[i] = common::initialize_random ( epsilon );

  for (int i = 0; i < M*N; i++)
    h_C[i] = common::initialize_random ( epsilon );
 
  BENCHMARK_ADVANCED("cuBLAS SGEMM")(Catch::Benchmark::Chronometer meter) {

    T *d_A, *d_B, *d_C;
    cudaMalloc( (void**) &d_A, sizeof(T) * M * K);
    cudaMalloc( (void**) &d_B, sizeof(T) * K * N);
    cudaMalloc( (void**) &d_C, sizeof(T) * M * N);

    stat = cublasCreate(&handle); // initialize CUBLAS context

    cudaMemcpy(d_A, h_A, M * K * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(T), cudaMemcpyHostToDevice);

    T alpha(1.0), beta(0.5);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    meter.measure( [handle, N, M, K, &alpha, &beta, d_B, d_A, d_C] { 
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N,
                     d_A, K, &beta, d_C, N); 
    cudaDeviceSynchronize() ;
    });
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msecTotal(0.0f);
    cudaEventElapsedTime(&msecTotal, start, stop);
    cudaMemcpy(h_C, d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost);

    double Gflops = 2.0 * M * N * K * 1e-9;
    double avg_time = msecTotal*1000000.0;
    //printf("Average elapsed time: (%7.6f) ns %f, %f, %f \n", avg_time, h_A[0], h_B[0], h_C[0]);
    //printf("Gflops = %f \n", Gflops);
    //printf("%f GFLOPs/s \n", Gflops/avg_time);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle); // destroy CUBLAS context
  };
  free(h_A);
  free(h_B);
  free(h_C);

  return h_A; 
}



} // namespace cuda_bench
