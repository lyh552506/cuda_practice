#include <cuda.h>
#include <stdlib.h>

#include "cute/tensor.hpp"
using namespace cute;
using namespace std;
#define PRINT(name, content) \
  print(name);               \
  print(" : ");              \
  print(content);            \
  print("\n");

#define PRINTTENSOR(name, content) \
  print(name);                     \
  print(" : ");                    \
  print_tensor(content);           \
  print("\n");


__global__ void handle_regiser_tensor() {
  auto rshape = make_shape(Int<4>{}, Int<2>{});
  auto rstride = make_stride(Int<2>{}, Int<1>{});
  auto rlayout = make_layout(rshape, rstride);
  auto rtensor = make_tensor(rlayout);
  print(rtensor);
}



int main() {
  // register tensor
  handle_regiser_tensor<<<1, 4>>>();
}