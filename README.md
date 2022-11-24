

To compile
module load git/2.11.1 cmake/3.23.1 llvm/13.0.1

cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3"
cmake --build build --parallel 8
