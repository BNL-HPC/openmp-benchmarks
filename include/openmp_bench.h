#pragma once

#include <iostream>
#include <omp.h>

namespace openmp_bench {

template <typename T>
void set_to_zero( T* device_array, const int N, const int blocksize, const int nblocks );

template <typename T>
T* set_to_zero_wrapper( const int N, const int blocksize );

template <typename T>
void collect_positive_devc ( T* devc_array, T* devc_array_positive, std::size_t* devc_count, const std::size_t N, const std::size_t nblocks, const std::size_t blocksize );

template <typename T>
T* atomic_capture_wrapper ( const std::size_t N, const std::size_t blocksize );

template <typename T>
T* atomic_update_wrapper ( const std::size_t N, const std::size_t blocksize );

template <typename T>
T* saxpy_wrapper ( const std::size_t N, const std::size_t blocksize );

void check_target_device () ;




} // namespace openmp_bench
