stencil_bench: main.cu stencil_kernels.h tools.h
	nvcc -arch=sm_60 -std=c++11 main.cu $(EXTRA_FLAGS) -o $@

clean: 
	rm -f stencil_bench
