#include <cuda.h>
#include <mma.h>
#include <stdlib.h>

#include <iostream>

#include "../3rdparty/cutlass/include/cute/tensor.hpp"
#include "../utils/helper.hpp"
using namespace cute;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, typename TiledMMA>
__global__ void CuTeGEMM(half *A, half *B, half *C, int M, int N, int K) {
  Tensor gA =
      make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
  Tensor gB =
      make_tensor(make_gmem_ptr(B), make_shape(K, N), make_stride(N, Int<1>{}));
  Tensor gC =
      make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));

  Tensor load_gA = local_tile(gA, make_tile(Int<BLOCK_M>{}, Int<BLOCK_K>{}),
                              make_coord(blockIdx.y, _));
  Tensor load_gB = local_tile(gB, make_tile(Int<BLOCK_N>{}, Int<BLOCK_K>{}),
                              make_coord(_, blockIdx.x));
  Tensor load_gC = local_tile(gC, make_tile(Int<BLOCK_M>{}, Int<BLOCK_N>{}),
                              make_coord(blockIdx.y, blockIdx.x));

  TiledMMA tiledmma;
  ThrMMA thr_mma = tiledmma.get_thread_slice(threadIdx.x);
  Tensor tAgA = thr_mma.partition_A(load_gA);
  Tensor tBgB = thr_mma.partition_B(load_gB);
  Tensor tCgC = thr_mma.partition_C(load_gC);

  Tensor tArA = thr_mma.partition_fragment_A(gA(_, _, 0));
  Tensor tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  Tensor tCrC = thr_mma.partition_fragment_C(gC(_, _));

  int slice_k = size<2>(load_gA);
  for (int i = 0; i < slice_k; i++) {
    copy(tAgA(_, _, _, i), tArA);
    copy(tBgB(_, _, _, i), tBrB);

    gemm(tiledmma, tCrC, tArA, tBrB, tCrC);
  }
  copy(tCrC, tCgC);
}

int main() {
  half *d_C, *d_A, *d_B, *A, *B, *C;

  constexpr int m = 1024 * 64;
  constexpr int n = 128;
  constexpr int k = 1024;

  constexpr int block_m = 128;
  constexpr int block_n = 64;
  constexpr int block_k = 128;
  cudaError_t err;
  checkCudaErrors(cudaMalloc(&d_C, sizeof(half) * m * n));
  checkCudaErrors(cudaMalloc(&d_A, sizeof(half) * m * k));
  checkCudaErrors(cudaMalloc(&d_B, sizeof(half) * k * n));

  A = (half *)malloc(sizeof(half) * m * k);
  B = (half *)malloc(sizeof(half) * n * k);
  C = (half *)malloc(sizeof(half) * m * n);

  helper::genRandomMatrix(A, m, k);
  helper::genRandomMatrix(B, k, n);
  helper::genEmptyMatrix(C, m, n);

  checkCudaErrors(
      cudaMemcpy(d_A, A, sizeof(half) * m * k, cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_B, B, sizeof(half) * n * k, cudaMemcpyHostToDevice));

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using MMA_Atom = MMA_Atom<mma_traits>;

  using TiledMMA = TiledMMA<MMA_Atom, Layout<Shape<Int<4>, _1, _1>>,
                            Tile<Int<16 * 4>, _16, _16>>;

  print(size(TiledMMA{}));

  dim3 Grid(n / block_n, m / block_m);
  dim3 Block(size(TiledMMA{}));

  CuTeGEMM<block_m, block_n, block_k, TiledMMA>
      <<<Grid, Block>>>(d_A, d_B, d_C, m, n, k);
}
