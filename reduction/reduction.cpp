#define CATCH_CONFIG_ENABLE_BENCHMARKING
#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING 
#endif

#include <iostream>
#include <type_traits>
#include <cstdlib>
#include <cmath>
#include <catch.hpp>
#include <openmp_bench.h>
#include <omp.h>

namespace openmp_bench {

template double* reduction_wrapper <double> ( const std::size_t, const std::size_t );
template float*  reduction_wrapper <float>  ( const std::size_t, const std::size_t );
template int*    reduction_wrapper <int>    ( const std::size_t, const std::size_t );

template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type initialize_random(T epsilon) 
{
  return (rand() % 200) - 100;
}

template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type initialize_random(T epsilon)
{
  return 2.0 * (rand() / static_cast <T> (RAND_MAX)) - 1.0;
}

template <typename T>
void reduction_kernel ( T* sum_d, T* data_d, T init_val, const int N, const int nblocks, const int blocksize ) 
{
#pragma omp target	
  sum_d[0] = init_val;
#pragma omp target teams distribute parallel for num_teams(nblocks) num_threads(blocksize) reduction(+:sum_d[0])
  for(int i=0; i<N; i++) 
  {
    sum_d[0] += data_d[i];
  }
  return; 
}

template <typename T>
T* reduction_wrapper ( const std::size_t N, const std::size_t blocksize ) 
{
  bool is_target_initial_device = false;	
#pragma omp target map(tofrom: is_target_initial_device)
  if (omp_is_initial_device ()) {
    printf( "Target region being executed on host!! Aborting!!!! \n");
    is_target_initial_device = true;
  }
  if ( is_target_initial_device )
    std::abort ();

  const std::size_t threads_tot = N;
  const std::size_t nblocks     = ( threads_tot + blocksize - 1 ) / blocksize;

  const T epsilon = 0.49;
  T* data = ( T* ) malloc( sizeof( T ) * N );
  T* sum = ( T* ) malloc( sizeof( T ) * 1 );
  T init_val;

  BENCHMARK_ADVANCED("OpenMP reduction")(Catch::Benchmark::Chronometer meter) {
    for(int i=0; i<N; i++)
    {
      data[i] = initialize_random ( epsilon );
    }
    init_val = initialize_random ( epsilon );
#pragma omp target enter data map(to:sum[0:1])
#pragma omp target enter data map(to:data[0:N])    
    //BENCHMARK ("OpenMP reduction") { return reduction_kernel ( result, data_x_device, data_y_device, fact, N, nblocks, blocksize); };
    meter.measure( [sum, data, init_val, N, nblocks, blocksize] 
    { return reduction_kernel ( sum, data, init_val, N, nblocks, blocksize); });
#pragma omp target exit data map(from:sum[0:1])
#pragma omp target exit data map(release:data[0:N])

    T correct_sum = init_val;
    for(int i = 0; i < N; i++) 
    {
      correct_sum += data[i];
    }
    CHECK ( std::fabs( correct_sum - sum[0] ) <= epsilon );
  };
  return sum; 
}

} //namespace openmp_bench
