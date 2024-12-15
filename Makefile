run:
	echo $(path)
	nvcc -arch=sm_89 -g -o test $(path)

	