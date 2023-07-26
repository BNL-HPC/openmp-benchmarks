

To compile
========
## BNL IC
========

module load git/2.11.1 cmake/3.23.1 llvm/13.0.1

cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3"

cmake --build build --parallel 8

========
## lambda2
========

module load nvhpc/22.9
export PATH=/work/software/wc/llvm-16-test/bin/:$PATH
export LD_LIBRARY_PATH=/work/software/wc/llvm-16-test/lib/:$LD_LIBRARY_PATH

/work/atif/packages/cmake-3.25.0-linux-x86_64/bin/cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -mtune=native" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_PREFIX_PATH=""

/work/atif/packages/cmake-3.25.0-linux-x86_64/bin/cmake --build build --parallel 8

./build/src/cuda_app --benchmark-samples 1000 --benchmark-resamples 100 --benchmark-confidence-interval 0.95 --input-file inputfile --benchmark-warmup-time
1

========
## perlmutter
========
module purge
module load llvm/16 cudatoolkit/11.7
cmake -S . -B build-llvm16-perlmutter -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="-O3 -mtune=native"
cmake --build
salloc
./app
Currently Loaded Modules:
  1) craype-x86-milan     4) xpmem/2.5.2-2.4_3.48__gd0f7936.shasta   7) cray-libsci/23.02.1.1  10) gcc/11.2.0              13) xalt/2.10.2              16) cudatoolkit/11.7
  2) libfabric/1.15.2.0   5) PrgEnv-gnu/8.3.3                        8) cray-mpich/8.1.25      11) perftools-base/23.03.0  14) Nsight-Compute/2022.1.1  17) craype-accel-nvidia80
  3) craype-network-ofi   6) cray-dsmml/0.2.2                        9) craype/2.7.20          12) cpe/23.03               15) Nsight-Systems/2022.2.1  18) gpu/1.0
cmake -S . -B build-clang15-perlmutter -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_FLAGS="-O3 -mtune=native --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/ --cuda-gpu-arch=sm_80 -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/include/ -L/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/lib64 -lcudart_static -ldl -lrt -pthreads" -DCMAKE_PREFIX_PATH="" -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/bin/nvcc -DCUDA_TOOLKIT_ROOT_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/
Jul 18, 2023
cmake -S . -B build-clang16-perlmutter -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="-O3 -mtune=native -L/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/lib64 -lcudart -lcudart_static -ldl -lrt -pthreads"
cmake --build build-clang15-perlmutter --parallel 16 --verbose



