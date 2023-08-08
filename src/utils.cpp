#include <openmp_bench.h>
#include <omp.h>

void openmp_bench::check_target_device () {

  bool is_target_initial_device = false;	
  #pragma omp target map(tofrom: is_target_initial_device)
  {
    if (omp_is_initial_device ()) {
      printf( "\nTarget region being executed on host!! Aborting!!!! \n Set OMP_TARGET_OFFLOAD=mandatory and try again \n");
      is_target_initial_device = true;
    } 
  }
  if ( is_target_initial_device )
    std::abort ();
  else
    printf( "\nTarget region being executed on device ID %d \n", omp_get_default_device() );

}

