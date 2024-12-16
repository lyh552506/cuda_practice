run:
	@echo $(path)
	nvcc -arch=sm_86 -g -o target $(path) -lcublas

	