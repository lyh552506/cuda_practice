THUNDERKITTENS_ROOT:=/root/autodl-tmp/cuda_practice/3rdparty/ThunderKittens
CuTe_ROOT:=/root/autodl-tmp/cuda_practice/3rdparty/cutlass/include
# ARCH:=-DKITTENS_HOPPER
ARCH:=-DKITTENS_4090
run:
	@echo $(path)
	nvcc -std=c++20 -g -G -o target $(path) --expt-relaxed-constexpr -lrt -lpthread -ldl ${ARCH} -arch=sm_89 -lcuda -lcudadevrt -lcudart_static -lcublas -I${CuTe_ROOT} -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype 

	