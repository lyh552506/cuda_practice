THUNDERKITTENS_ROOT:=/root/autodl-tmp/cuda_practice/3rdparty/ThunderKittens
CuTe_ROOT:=/root/autodl-tmp/cuda_practice/3rdparty/cutlass/include
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -g -G -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -MD -MT -MF -x cu -lrt -lpthread -ldl -lcuda
# ARCH:=-DKITTENS_HOPPER
ARCH:=-DKITTENS_4090
run:
	@echo $(path)
	nvcc -std=c++20 -g -G -o target $(path) ${ARCH} -arch=sm_89 $(NVCCFLAGS) -I${CuTe_ROOT} -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype 

	