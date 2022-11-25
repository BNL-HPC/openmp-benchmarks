#include <openmp_bench.h>
#include <omp.h>

int main()
{
  openmp_bench <double, 4096, 256> x;
  x.set_array_zero ( );

  return 0;
}
