#include "../3rdparty/ThunderKittens/include/kittens.cuh"
#include "../utils/helper.hpp"
using namespace kittens;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 32;

constexpr int NUM_WORKERS = 4;
constexpr int PIPE_STAGES = 2;
constexpr int LOAD_GROUPS = 2;
constexpr int BLOCK_SIZE = NUM_WORKERS * WARP_THREADS;

using shared_tileA = st_bf<BLOCK_M, BLOCK_K>;
using shared_tileB = st_bf<BLOCK_K, BLOCK_N>;
using shared_tileC = st_bf<BLOCK_M, BLOCK_N>;

using reg_tileA = rt_bf<BLOCK_M, BLOCK_K>;
using reg_tileB =
    rt_bf<BLOCK_K, BLOCK_N,
          kittens::ducks::rt_layout::col>;  // this has to be col major
using reg_tileC = rt_fl<BLOCK_M, BLOCK_N>;

template <int M, int K>
using a_gl = gl<bf16, 1, 1, M, K, shared_tileA>;
template <int K, int N>
using b_gl = gl<bf16, 1, 1, K, N, shared_tileB>;
template <int M, int N>
using c_gl = gl<bf16, 1, 1, M, N, shared_tileC>;

template <int M, int N, int K>
struct gemm_globals {
  a_gl<M, K> a;
  b_gl<K, N> b;
  c_gl<M, N> c;
};

template <int M, int N, int K>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void gemm(
    const __grid_constant__ gemm_globals<M, N, K> g) {
  using load_group = kittens::group<(NUM_WORKERS / LOAD_GROUPS)>;

  auto workerid = kittens::warpid();

  auto row_worker = workerid / 2;
  auto col_worker = workerid % 2;

  auto load_id = load_group::groupid();

  constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;

  int warp_row = LOAD_GROUPS * blockIdx.y;
  int warp_col = LOAD_GROUPS * blockIdx.x;

  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int *)&__shm[0]);

  shared_tileA(&a_s)[LOAD_BLOCKS][PIPE_STAGES] =
      al.allocate<shared_tileA, LOAD_BLOCKS, PIPE_STAGES>();
  shared_tileB(&b_s)[LOAD_BLOCKS][PIPE_STAGES] =
      al.allocate<shared_tileB, LOAD_BLOCKS, PIPE_STAGES>();

  shared_tileC(&c_s)[LOAD_BLOCKS][LOAD_BLOCKS] =
      reinterpret_cast<shared_tileC(&)[LOAD_BLOCKS][LOAD_BLOCKS]>(a_s);

  reg_tileA ar_bf;
  reg_tileB br_bf;
  reg_tileC cr_fl;

  zero(cr_fl);
  int numKtile = K / BLOCK_K;
  int tic = 0;

  load_group::load_async<2, true>(a_s[load_id][tic], g.a,
                                  {warp_row + load_id, 0});
  load_group::load_async<2, true>(b_s[load_id][tic], g.b,
                                  {0, warp_col + load_id});

  for (int inner = 0; inner < numKtile;
       inner++, tic = (tic + 1) % PIPE_STAGES) {
    int next_load_idx = inner + 1;
    if (next_load_idx < numKtile) {
      int next_tic = (tic + 1) % PIPE_STAGES;
      load_group::load_async<2, true>(a_s[load_id][next_tic], g.a,
                                      {warp_row + load_id, next_load_idx});
      load_group::load_async<2, true>(b_s[load_id][next_tic], g.b,
                                      {next_load_idx, warp_col + load_id});
      load_async_wait<2>();
    } else
      load_async_wait();

    __syncthreads();

    load(ar_bf, a_s[row_worker][tic]);
    load(br_bf, b_s[col_worker][tic]);
    mma_AB(cr_fl, ar_bf, br_bf, cr_fl);
  }

  __syncthreads();

  store(c_s[row_worker][col_worker], cr_fl);
  store<2, false>(g.c, c_s[row_worker][col_worker],
                  {warp_row + row_worker, warp_col + col_worker});
}

int main() {
  const size_t M = 4096;
  const size_t K = 4096;
  const size_t N = 4096;

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

  dim3 grid((N, BLOCK_N / LOAD_GROUPS), (M, BLOCK_M / LOAD_GROUPS));
  dim3 block(BLOCK_SIZE);
  a_gl<M, K> a_arg{dA, nullptr, nullptr, nullptr, nullptr};
  b_gl<K, N> b_arg{dB, nullptr, nullptr, nullptr, nullptr};
  c_gl<M, N> c_arg{dC, nullptr, nullptr, nullptr, nullptr};
  gemm_globals<M, N, K> g(a_arg, b_arg, c_arg);

  gemm<M, N, K><<<grid, block>>>(g);

  cudaMemcpy(C, dC, M * N * sizeof(bf16), cudaMemcpyDeviceToHost);

  delete[] A;
  delete[] B;
  delete[] C;

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}
