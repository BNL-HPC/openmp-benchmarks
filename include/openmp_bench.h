#pragma once

#include <iostream>
#include <omp.h>

namespace openmp_bench {

template <typename T>
void set_to_zero( T* device_array, const int N, const int blocksize, const int nblocks );

template <typename T>
T* set_to_zero_wrapper( const int N, const int blocksize );

} // namespace openmp_bench
