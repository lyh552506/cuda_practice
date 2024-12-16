#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <type_traits>

#include "../utils/helper.hpp"

__global__ void naivesgemm(float *a, float *b, float *c, const int M,
                           const int N, const int K) {
  auto tx = blockDim.x * blockIdx.x + threadIdx.x;
  auto ty = blockDim.y * blockIdx.y + threadIdx.y;
  if (tx < M && ty < N) {
    float acc = 0;
	#pragma unroll
    for (int i = 0; i < K; i++) {
      acc += a[tx * M + i] * b[i * N + ty];
    }
    c[tx * M + ty] = acc;
  }
}

int main() {
  int M = 1024, N = 1024, K = 512;
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
  helper::genRandomMatrix(a, M, K,0);
  helper::genRandomMatrix(b, K, N,1);
  helper::genEmptyMatrix(c, M, N);
  checkCudaErrors(cudaMemcpy(d_a, a, ASIZE(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b, b, BSIZE(float), cudaMemcpyHostToDevice));

  dim3 Grid((M + 31) / 32, (N + 31) / 32, 1);
  dim3 Block(32, 32, 1);
  naivesgemm<<<Grid, Block>>>(d_a, d_b, d_c, M, N, K);
//   cublasSgemm()
  checkCudaErrors(cudaMemcpy(c, d_c, CSIZE(float), cudaMemcpyDeviceToHost));
  helper::printMatrix(c, M, N);
}