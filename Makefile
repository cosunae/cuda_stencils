stencil_bench: main.cu stencil_kernels.h tools.h
	nvcc -arch=sm_60 -std=c++11 -I./libjson -DJSON_ISO_STRICT libjson/libjson.a main.cu -O3 -o $@

clean: 
	rm -f stencil_bench
