#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "../utils/helper.hpp"

__global__ void Sgemm_v1(float *a, float *b, float *c, const int M, const int N,
                         const int K) {
  const int bm = 128, bn = 128, bk = 8, tm = 8, tn = 8;
  // prepare smem
  __shared__ float smem_a[bm][bk];
  __shared__ float smem_b[bk][bn];
  // prepare acc reg
  float acc[tm][tn] = {0.0};
  // calc real index using thread info
  auto index = blockDim.x * threadIdx.y + threadIdx.x;
  // calc thread load ind
  auto load_a_smem_m = index / 2;
  auto load_a_smem_k = (index % 2) * 4;
  auto load_b_smem_k = index / 32;
  auto load_b_smem_n = (index % 32) * 4;

  for (auto blocking_k_ind = 0; blocking_k_ind < (K + bk - 1) / bk;
       blocking_k_ind++) {
    // calc correspond addr in gmem
    auto load_a_gmem_k = blocking_k_ind * bk + load_a_smem_k;
    auto load_a_gmem_m = blockIdx.y * bm + load_a_smem_m;
    auto load_b_gmem_k = blocking_k_ind * bk + load_b_smem_k;
    auto load_b_gmem_n = blockIdx.x * bn + load_b_smem_n;
    // gmem to smem
    FLOAT4(smem_a[load_a_smem_m][load_a_smem_k]) =
        FLOAT4(a[load_a_gmem_m * K + load_a_gmem_k]);
    FLOAT4(smem_b[load_b_smem_k][load_b_smem_n]) =
        FLOAT4(b[load_b_gmem_k * N + load_b_gmem_n]);
    __syncthreads();

    // do blocked matmul
#pragma unroll
    for (int i = 0; i < bk; i++) {
#pragma unroll
      for (int j = 0; j < tm; j++) {
#pragma unroll
        for (int h = 0; h < tn; h++) {
          auto calc_a_addr_m = threadIdx.y * tm + j;
          auto calc_b_addr_n = threadIdx.x * tn + h;
          acc[j][h] += smem_a[calc_a_addr_m][i] * smem_b[i][calc_b_addr_n];
        }
      }
    }
    __syncthreads();
  }

// store to gmem
#pragma unroll
  for (int i = 0; i < tm; i++) {
    auto correspond_gmem_m = blockIdx.y * bm + threadIdx.y * tm + i;
#pragma unroll
    for (int j = 0; j < tn; j += 4) {
      auto correspond_gmem_n = blockIdx.x * bn + threadIdx.x * tn + j;
      FLOAT4(c[correspond_gmem_m * N + correspond_gmem_n]) = FLOAT4(acc[i][j]);
    }
  }
}

int main() {
  int M = 1024, N = 1024, K = 512;
  int bm = 128, bn = 128, bk = 8;
  int tm = 8, tn = 8;
  float *a, *b, *c;
  a = (float *)malloc(ASIZE(float));
  b = (float *)malloc(BSIZE(float));
  c = (float *)malloc(CSIZE(float));

  //   helper::printMatrix(a, M, K);
  //   helper::printMatrix(b, K, N);
  //   helper::printMatrix(c, M, K);
  float *d_a, *d_b, *d_c;
  checkCudaErrors(cudaMalloc(&d_a, ASIZE(float)));
  checkCudaErrors(cudaMalloc(&d_b, BSIZE(float)));
  checkCudaErrors(cudaMalloc(&d_c, CSIZE(float)));
  helper::genRandomMatrix(a, M, K);
  helper::genRandomMatrix(b, K, N);
  helper::genEmptyMatrix(c, M, N);
  checkCudaErrors(cudaMemcpy(d_a, a, ASIZE(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b, b, BSIZE(float), cudaMemcpyHostToDevice));

  dim3 Grid((M + bm - 1) / bm, (N + bn - 1) / bn, 1);
  dim3 Block(bm / tm, bn / tn, 1);
  Sgemm_v1<<<Grid, Block>>>(d_a, d_b, d_c, M, N, K);
  checkCudaErrors(cudaMemcpy(c, d_c, CSIZE(float), cudaMemcpyDeviceToHost));
  helper::printMatrix(c, M, N);
}