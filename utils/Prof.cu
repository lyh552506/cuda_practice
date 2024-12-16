#include <driver_types.h>
#include <vector_types.h>

template <typename tp>
int Performance(void (*kernel)(tp *, tp *, tp *, const int, const int,
                               const int),
                dim3 Grid, dim3 Block, int M, int N, int K, int repeat) {
  tp *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, sizeof(tp) * M * K);
  cudaMalloc(&d_b, sizeof(tp) * K * N);
  cudaMalloc(&d_c, sizeof(tp) * M * N);

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);
  for (int i = 0; i < repeat; i++)
    kernel<<<Grid, Block>>>(d_a, d_b, d_c, M, N, K);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float msec, sec;
  cudaEventElapsedTime(&msec, beg, end);
  sec = msec / 1000.0 / repeat;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return sec;
}