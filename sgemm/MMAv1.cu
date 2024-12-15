#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <type_traits>

#include "../utils/helper.hpp"

__global__ void Sgemm_v1(float *a, float *b, float *c, const int M, const int N,
                         const int K, const int bm, const int bn, const int bk,
                         const int tm, const int tn) {
  auto bx = blockDim.x;
}

int main() {
  int M = 128, N = 128, K = 64;
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
  Sgemm_v1<<<Grid, Block>>>(d_a, d_b, d_c, M, N, K, bm, bn, bk, tm, tn);
  checkCudaErrors(cudaMemcpy(c, d_c, CSIZE(float), cudaMemcpyDeviceToHost));
  helper::printMatrix(c, M, N);
}