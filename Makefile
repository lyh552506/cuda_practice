run:
	@echo $(path)
	nvcc -arch=sm_80 -std=c++17 -g -o target $(path) -I3rdparty/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none 

	