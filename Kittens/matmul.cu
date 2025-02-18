#include <cstddef>

#include "../3rdparty/ThunderKittens/include/kittens.cuh"
#include "../utils/helper.hpp"
using namespace kittens;

constexpr int NUM_WORKERS = 4;
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;
constexpr int block_m = 64;
constexpr int block_n = 64;
constexpr int block_k = 64;

constexpr int Load_Group = 2;

template <typename T>
using global_layout = gl<T, 1, 1, -1, -1>;
template <typename T>
struct globals {
  global_layout<T> a, b, c;
};

__launch_bounds__(NUM_WORKERS* WARP_THREADS, 1) __global__
    void matmul(const bf16* A, const bf16* B, bf16* C, const globals<bf16>& g) {
		
}

int main() {
  bf16 *A, *B, *C, *dA, *dB, *dC;
  A = new bf16[M * K];
  B = new bf16[K * N];
  C = new bf16[M * N];
  helper::genRandomMatrix(A, M, K);
  helper::genRandomMatrix(B, K, N);
  helper::genEmptyMatrix(C, M, N);

  cudaMalloc(&dA, M * K * sizeof(bf16));
  cudaMalloc(&dB, K * N * sizeof(bf16));
  cudaMalloc(&dC, M * N * sizeof(bf16));

  cudaMemcpy(dA, A, M * K * sizeof(bf16), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, K * N * sizeof(bf16), cudaMemcpyHostToDevice);

  global_layout<bf16> a(dA, nullptr, nullptr, M, K);
  global_layout<bf16> b(dB, nullptr, nullptr, K, N);
  global_layout<bf16> c(dC, nullptr, nullptr, M, N);
  globals<bf16> g{a, b, c};

  dim3 Grid((M + block_m * Load_Group - 1) / block_m * Load_Group,
            (N + block_n * Load_Group - 1) / block_n * Load_Group);
  dim3 Block(NUM_WORKERS * WARP_THREADS);

  matmul<<<Grid, Block>>>(dA, dB, dC, g);
}