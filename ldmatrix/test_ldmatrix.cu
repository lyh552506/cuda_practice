#include "cublas_v2.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <type_traits>

#define M 4
#define N 8
#define K 4

template <typename tp> __device__ void PrintMatrix(tp *arr, int row, int col) {
  //   auto arr_pret = reinterpret_cast<uint8_t *>(arr);
  for (int i = 0; i < row; i++) {
    printf("Matrix row: %d ", i);
    for (int j = 0; j < col; j++) {
      if constexpr (std::is_same<tp, int>::value) {
        printf("%d(row:%d) ", arr[i * col + j], j);
      } else if constexpr (std::is_same<tp, half>::value) {
        printf("%.0f(row:%d) ", __half2float(arr[i * col + j]), j);
      } else {
        assert(0);
      }
    }
    printf("\n");
  }
}

/// With No .trans, use i32 smem type to handle b16
__global__ void TestLdMatrix_i32() {
  __shared__ uint32_t smem[M * N * K];
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < 4 * 8 * 4; ++i) {
      smem[i] = i;
    }
  }
  __syncthreads();
  if (tid == 0)
    PrintMatrix(smem, 16, 8);
  int aTile_index = tid % 16 * 8 + tid / 16 * 4;
  uint32_t a[4];
  uint32_t smem_ptr = __cvta_generic_to_shared(smem + aTile_index);
  //   printf("Tid : %d ,Smem_ptr : %d\n", tid, smem_ptr);
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4];\n"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(smem_ptr));
  printf("Tid : %d ,Ldmatrix val : %d %d %d %d\n", tid, a[0], a[1], a[2], a[3]);
}

/// With No .trans, use b16 smem type to handle b16
__global__ void TestLdMatrix_b16() {
  __shared__ half smem[4 * 8 * 8];
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < 4 * 8 * 8; ++i) {
      smem[i] = i;
    }
  }
  __syncthreads();
  if (tid == 0)
    PrintMatrix(smem, 16, 16);
  int aTile_index = tid % 16 * 16 + tid / 16 * 8;
  uint32_t a[4];
  uint32_t smem_ptr = __cvta_generic_to_shared(smem + aTile_index);
  //   printf("Tid : %d ,Smem_ptr : %d\n", tid, smem_ptr);
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4];\n"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(smem_ptr));
  auto tmp = reinterpret_cast<half *>(a);
  printf("Tid : %d ,Ldmatrix val : %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
         tid, __half2float(tmp[0]), __half2float(tmp[1]), __half2float(tmp[2]),
         __half2float(tmp[3]), __half2float(tmp[4]), __half2float(tmp[5]),
         __half2float(tmp[6]), __half2float(tmp[7]));
}

/// Use .trans, use i32 smem type to handle b16
__global__ void TestLdMatrix_Trans_b16() {
  __shared__ half smem[2 * 8 * 8];
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < 2 * 8 * 8; ++i) {
      smem[i] = i;
    }
  }
  __syncthreads();
  if (tid == 0)
    PrintMatrix(smem, 16, 8);
  int aTile_index = tid * 8;
  uint32_t a[2];
  uint32_t smem_ptr = __cvta_generic_to_shared(smem + aTile_index);
  printf("Tid : %d ,Smem_ptr : %d\n", tid, smem_ptr);
  asm("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 { %0, %1}, [ %2];\n"
      : "=r"(a[0]), "=r"(a[1])
      : "r"(smem_ptr));
  auto tmp = reinterpret_cast<half *>(a);
  printf("Tid : %d ,Ldmatrix val : %.0f %.0f %.0f %.0f\n", tid,
         __half2float(tmp[0]), __half2float(tmp[1]), __half2float(tmp[2]),
         __half2float(tmp[3]));
}

int main() {
  dim3 Block(32, 1, 1);
  dim3 Grid(1, 1, 1);
  //   TestLdMatrix_i32<<<Grid, Block>>>();
  TestLdMatrix_b16<<<Grid, Block>>>();
  //   TestLdMatrix_Trans_b16<<<Grid, Block>>>();
  cudaDeviceReset();
  return 0;
}
