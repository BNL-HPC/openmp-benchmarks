

To compile
module load git/2.11.1 cmake/3.23.1 llvm/13.0.1

cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3"
cmake --build build --parallel 8

lambda2
/work/atif/packages/cmake-3.25.0-linux-x86_64/bin/cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -mtune=native" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_PREFIX_PATH=""
/work/atif/packages/cmake-3.25.0-linux-x86_64/bin/cmake --build build --parallel 8
