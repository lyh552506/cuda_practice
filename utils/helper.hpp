#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>

#define ASIZE(type) sizeof(type) * M* K
#define BSIZE(type) sizeof(type) * K* N
#define CSIZE(type) sizeof(type) * M* N

#define checkCudaErrors(func)                                                \
  {                                                                          \
    auto e = (func);                                                         \
    if (e != cudaSuccess)                                                    \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  }

namespace helper {
void genRandomMatrix(float* A, int M, int N) {
  srand(0);  // Initialization, should only be called once.
  float a = 5.0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = (float)rand() / ((float)RAND_MAX / a);
    }
  }
}

void genEmptyMatrix(float* A, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) A[i * N + j] = 0;
  }
}

void printMatrix(float* A, int M, int N) {
  for (int i = 0; i < M; i++) {
    // std::cout << "Row:" << i << " ";
    for (int j = 0; j < N; j++) {
      std::cout << A[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}
};  // namespace helper
