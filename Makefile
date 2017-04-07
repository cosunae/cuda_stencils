all: main.cu
	nvcc -arch=sm_20 -std=c++11 $^ $(EXTRA_FLAGS) -o $@

