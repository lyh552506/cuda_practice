cmake_minimum_required(VERSION 3.18)
project(cuda_practice LANGUAGES CXX CUDA)

# 设置 CUDA 编译器
set(CMAKE_CUDA_COMPILER nvcc)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -g -G --expt-relaxed-constexpr -arch=sm_89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcuda -lcudadevrt -lcudart_static -lcublas -lrt -lpthread -ldl")

# 设置包含目录
set(CuTe_ROOT /root/autodl-tmp/cuda_practice/3rdparty/cutlass/include)
include_directories(${CuTe_ROOT})

# 设置 ARCH 选项
# set(ARCH -DKITTENS_4090)
# add_definitions(${ARCH})

# 添加源文件
set(path /root/autodl-tmp/cuda_practice/CuTe/mma.cu)
set(SOURCE_FILES ${path})

# 添加可执行文件
add_executable(target ${SOURCE_FILES})