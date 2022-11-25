#pragma once

#include <iostream>
#include <omp.h>

template<typename T, int size, int blocksize>
class openmp_bench
{
public:

  openmp_bench();
  ~openmp_bench();
 
  void set_array_zero( );

//private:
  T * m_device_array;
  const int m_threads_tot    = size;
  const int m_blocksize      = blocksize;
  const int m_nblocks        = ( m_threads_tot + m_blocksize - 1 ) / m_blocksize;
  const std::size_t m_offset = 0;
  int m_default_device;
  int m_initial_device;
};
	
template<typename T, int size, int blocksize>
openmp_bench<T, size, blocksize>::openmp_bench() {

  m_default_device = omp_get_default_device();
  m_initial_device = omp_get_initial_device();
 
  /* Allocate array of length N on target */
  m_device_array = (T*) omp_target_alloc( m_threads_tot * sizeof( T ), m_default_device);
  if ( m_device_array == NULL ) {
    std::cout << " ERROR: No space left on device." << std::endl;
  }
}

template<typename T, int size, int blocksize>
openmp_bench<T, size, blocksize>::~openmp_bench() {

  omp_target_free( m_device_array, m_default_device);
   
}

template<typename T, int size, int blocksize>
inline void openmp_bench<T, size, blocksize>::set_array_zero( )
{
  auto device_array = m_device_array;	
  int i;

  #pragma omp target is_device_ptr ( device_array )            
  #pragma omp teams distribute parallel for num_teams(m_nblocks) num_threads(m_blocksize) 
  for(i = 0; i < m_threads_tot; i++) {
    //printf(" num teams = %d, num threads = %d", omp_get_num_teams(), omp_get_num_threads() );
    m_device_array[i] = 0.;
  }

  return;
}

// Explicitly instantiate only the classes you want to be defined.
//template class openmp_bench<double, 4096, 256>;
//template class openmp_bench<double, 4096, 256>;
//template class openmp_bench<double, 4096, 256>;
