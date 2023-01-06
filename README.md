

To compile
========
BNL IC
========

module load git/2.11.1 cmake/3.23.1 llvm/13.0.1

cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3"

cmake --build build --parallel 8

========
lambda2
========

module load nvhpc/22.9
export PATH=/work/software/wc/llvm-16-test/bin/:$PATH
export LD_LIBRARY_PATH=/work/software/wc/llvm-16-test/lib/:$LD_LIBRARY_PATH

/work/atif/packages/cmake-3.25.0-linux-x86_64/bin/cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -mtune=native" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_PREFIX_PATH=""

/work/atif/packages/cmake-3.25.0-linux-x86_64/bin/cmake --build build --parallel 8

./build/src/cuda_app --benchmark-samples 1000


export LD_LIBRARY_PATH=/work/atif/packages/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-9.3.0/gcc-12.2.0-qhtr62i46cvhlmyjtkawlijkkuc6cybu/lib64/:$LD_LIBRARY_PATH
export PATH=/work/atif/packages/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-9.3.0/gcc-12.2.0-qhtr62i46cvhlmyjtkawlijkkuc6cybu/bin/:$PATH
