#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "../utils/helper.hpp"

__global__ void naivesgemm(float *a, float *b, float *c, const int M,
                           const int N, const int K) {
  auto tx = blockDim.x * blockIdx.x + threadIdx.x;
  auto ty = blockDim.y * blockIdx.y + threadIdx.y;
  if (tx < M && ty < N) {
    int acc = 0;
	#pragma unroll
    for (int i = 0; i < K; i++) {
      acc += a[tx * M + i] * b[i * N + ty];
    }
    c[tx * M + ty] = acc;
  }
}

int main() {
  int M = 128, N = 128, K = 64;
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

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  float cublas_alpha = 1.0;
  float cublas_beta = 0;
  cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha,
              d_b, N, d_a, K, &cublas_beta, d_c, N);

  checkCudaErrors(cudaMemcpy(c, d_c, CSIZE(float), cudaMemcpyDeviceToHost));
  helper::printMatrix(c, M, N);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}