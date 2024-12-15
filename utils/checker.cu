#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    // 标准的每个 Block 的共享内存大小
    std::cout << "Max shared memory per block: " 
              << prop.sharedMemPerBlock << " bytes" << std::endl;

    // 查询支持的共享内存大小（可选分配）
    int sharedMemOptin = 0;
    cudaDeviceGetAttribute(&sharedMemOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    std::cout << "Max opt-in shared memory per block: " 
              << sharedMemOptin << " bytes" << std::endl;

    return 0;
}
