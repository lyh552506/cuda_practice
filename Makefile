run:
	@echo $(path)
	nvcc -std=c++20 -g -G -o target $(path) --expt-relaxed-constexpr -lrt -lpthread -ldl -DKITTENS_4090 -arch=sm_89 -lcuda -lcudadevrt -lcudart_static -lcublas 

	