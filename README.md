# Microbenchmarking OpenMP target offload with Catch2

## Build instructions for various machines

Currently for OpenMP target offload we need to edit the flag --offload-arch in CMakeLists.txt according to the targeted GPU architecture

After upgrading to Catch2 v3.x, specify path to Catch2 via -DCatch_Root=/path/to/Catch2, otherwise CMake will add it as a dependency

### BNL Institutional Cluster

Set --offload-arch=sm_37 for K80

Set --offload-arch=sm_60 for P100

module load git/2.11.1 cmake/3.23.1 llvm/13.0.1

cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3"

cmake --build build --parallel 8

### BNL CSI alpha1/lambda1/lambda2/lambda3/lambda4

Set --offload-arch=sm_80 for alpha1 A30

Set --offload-arch=sm_70 for lambda1 V100

Set --offload-arch=sm_86 for lambda2 A6000

Set --offload-arch=gfx906 for lambda2 Vega20

Set --offload-arch=gfx906 for lambda3

Set --offload-arch=sm_75 for lambda4 2080Ti

module use /work/software/modulefiles

module load nvhpc/22.9

export PATH=/work/software/wc/llvm-16-test/bin/:$PATH

export LD_LIBRARY_PATH=/work/software/wc/llvm-16-test/lib/:$LD_LIBRARY_PATH

/work/atif/packages/cmake-3.25.0-linux-x86_64/bin/cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -mtune=native" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_PREFIX_PATH=""

/work/atif/packages/cmake-3.25.0-linux-x86_64/bin/cmake --build build --parallel 8

### NERSC Perlmutter

Set --offload-arch=sm_80 for A100 

module load llvm/16 cudatoolkit/11.7 cmake/3.24.3

cmake -S . -B build-clang16-perlmutter -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="-O3 -mtune=native -L/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/lib64 -lcudart -lcudart_static -ldl -lrt -pthreads"

cmake --build build-clang16-perlmutter --parallel 16 --verbose

### OLCF Frontier

Set --offload-arch=gfx90a for Mi250X

module load rocm/5.4.3 cmake craype-accel-amd-gfx90a

cmake -S . -B build-clang15-frontier -DCMAKE_C_COMPILER=amdclang -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_CXX_FLAGS="-O3 -mtune=native " -DCMAKE_PREFIX_PATH=""

cmake --build build-clang15-frontier --parallel 16 --verbose

### BNL CSI HPC Dahlia

/home/atif/packages/cmake-3.30.0-rc2-linux-x86_64/bin/cmake -B build-dahlia/ -S . -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCatch2_ROOT=/home/atif/openmp-benchmarks/Catch22

/home/atif/packages/cmake-3.30.0-rc2-linux-x86_64/bin/cmake --build build-dahlia/ --parallel 16

./build/saxpy/saxpy_omp_app --benchmark-samples 1000 --benchmark-resamples 100 --benchmark-confidence-interval 0.95 --input-file inp-omp --benchmark-warmup-time 10 -r tabular

## Running a microbenchmark

./build/saxpy/saxpy_omp_app --benchmark-samples 1000 --benchmark-resamples 100 --benchmark-confidence-interval 0.95 --input-file inputfile --benchmark-warmup-time 10
