#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "../utils/helper.hpp"

__global__ void Sgemm_v2(float* a, float* b, float* c, const int M, const int N,
                         const int K) {

}


int main() {
  int M = 1024, N = 1024, K = 512;
  int bm = 128, bn = 128, bk = 8;
  int tm = 8, tn = 8;
  float *a, *b, *c;

  a = (float *)malloc(ASIZE(float));
  b = (float *)malloc(BSIZE(float));
  c = (float *)malloc(CSIZE(float));

  float *d_a, *d_b, *d_c;
  checkCudaErrors(cudaMalloc(&d_a, ASIZE(float)));
  checkCudaErrors(cudaMalloc(&d_b, BSIZE(float)));
  checkCudaErrors(cudaMalloc(&d_c, CSIZE(float)));
  helper::genRandomMatrix(a, M, K, 0);
  helper::genRandomMatrix(b, K, N, 1);
  helper::genEmptyMatrix(c, M, N);
  checkCudaErrors(cudaMemcpy(d_a, a, ASIZE(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b, b, BSIZE(float), cudaMemcpyHostToDevice));

  dim3 Grid((M + bm - 1) / bm, (N + bn - 1) / bn, 1);
  dim3 Block(bm / tm, bn / tn, 1);
  //   auto res = Performance<float>(Sgemm_v1, Grid, Block, M, N, K, 15);
  //   printf("res:%d", res);
  Sgemm_v2<<<Grid, Block>>>(d_a, d_b, d_c, M, N, K);
  checkCudaErrors(cudaMemcpy(c, d_c, CSIZE(float), cudaMemcpyDeviceToHost));
//   helper::printMatrix(c, M, N);
}