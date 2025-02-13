run:
	@echo $(path)
	nvcc -arch=sm_86 -std=c++17 -g -G -o target $(path) -I3rdparty/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none 

	