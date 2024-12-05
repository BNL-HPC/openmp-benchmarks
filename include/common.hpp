#pragma once

#include <iostream>
#include <cstdlib> 
#include <ctime> 

namespace common {

template <typename T>                                                
T initialize_random ( T epsilon ) {                                  

  if (std::is_same <float, T>::value) {                              
    return 2.0 * (rand() / static_cast <T> (RAND_MAX)) - 1.0;        
  }                                                                  
  if (std::is_same <double, T>::value) {                             
    return 2.0 * (rand() / static_cast <T> (RAND_MAX)) - 1.0;       
  }                                                                  
  if (std::is_same <int, T>::value) {                     
    return (rand() % 200) - 100;                                     
  }                                                                  
}

template <typename T> 
T get_epsilon () { return 1.0e-6; }
template <> double get_epsilon <double> () { return 1.0e-6; }
template <> float  get_epsilon <float>  () { return 1.0e-3; }
template <> int    get_epsilon <int>    () { return 0; }

template <typename T>
std::size_t collect_positive_serial_host ( T* host_array, T* host_array_positive, const std::size_t N ) {

  std::size_t host_ser_count = 0;

  for ( int i = 0; i < N; i++ ) {
      if ( host_array[i] > 0. ) {
        host_array_positive[host_ser_count] = host_array[i];
        host_ser_count++;
      }
  }

  return host_ser_count;	  
}



} // namespace common
