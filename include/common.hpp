#pragma once

#include <iostream>

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
template <> float  get_epsilon <float>  () { return 1.0e-2; }
template <> int    get_epsilon <int>    () { return 0; }


} // namespace common
