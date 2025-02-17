THUNDERKITTENS_ROOT:=/root/autodl-tmp/cuda_practice/3rdparty/ThunderKittens
run:
	@echo $(path)
	nvcc -std=c++20 -g -G -o target $(path) --expt-relaxed-constexpr -lrt -lpthread -ldl -DKITTENS_HOPPER -arch=sm_90a -lcuda -lcudadevrt -lcudart_static -lcublas -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype 

	