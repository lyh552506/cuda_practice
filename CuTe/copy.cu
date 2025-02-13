#include "../include/utils.hpp"

using namespace cute;
using namespace std;

int main() {
  using T = cute::half_t;

  // constexpr int M = 128;
  // constexpr int N = 32;
  //   constexpr int M = 128;
  //   constexpr int N = 128;

  //   using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  //   using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  //   using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

  //   using G2SCopy =
  //       decltype(make_tiled_copy(g2s_copy_atom{},
  //                                make_layout(make_shape(Int<32>{}, Int<4>{}),
  //                                            make_stride(Int<4>{},
  //                                            Int<1>{})),
  //                                make_layout(make_shape(Int<1>{},
  //                                Int<8>{}))));

  using MMA = MMA_Traits<SM80_16x8x8_F32F16F16F32_TN>;
  print("ALayout: ");
  print(typename MMA::ALayout{});
  print("\n");
  print("BLayout: ");
  print(typename MMA::BLayout{});
  print("\n");
  print("CLayout: ");
  print(typename MMA::CLayout{});
  print("\n");
}