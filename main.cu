#include <iostream>
#include <vector>
#include <chrono>

#define FNNAME(a) a
#define LOAD(a) __ldg(& a)

#include "stencil_kernels.h"

#undef FNNAME
#undef LOAD


#define FNNAME(a) a##_ldg
#define LOAD(a) a

#include "stencil_kernels.h"

#undef FNNAME
#undef LOAD


template<typename T> 
void launch() {

    const size_t isize=32;
    const size_t jsize=32;
    const size_t ksize=60;
    const size_t halo=2;
    const size_t alignment=32;
    const size_t right_padding=(isize+halo)%alignment;
    const size_t first_padding=alignment-halo;
    const size_t total_size=first_padding+(isize+right_padding)*(jsize+halo*2)*(ksize+halo*2);
    const size_t jstride = (isize+right_padding);
    const size_t kstride = jstride*jsize;

    const size_t init_offset = initial_offset(first_padding, halo, jstride, kstride);


    T* a;
    T* b;
    cudaMallocManaged(&a, sizeof(T)*total_size);
    cudaMallocManaged(&b, sizeof(T)*total_size);    


    for(size_t i=0; i < isize; ++i) {
        for(size_t j=0; j < jsize; ++j) {
            for(size_t k=0; k < ksize; ++k) {
                a[i+j*jstride + k*kstride + init_offset] = i+j*jstride + k*kstride + init_offset;
            }
        }
    }

    const size_t block_size_x = 32;
    const size_t block_size_y = 8;

    const size_t nbx = isize/block_size_x;
    const size_t nby = jsize/block_size_y;
   
    dim3 gd(nbx, nby,1);
    dim3 bd(block_size_x, block_size_y);

    std::vector<double> timings(5);

    std::chrono::high_resolution_clock::time_point t1,t2;

    for(size_t t=0; t < 10; t++) {
   
        
        t1 = std::chrono::high_resolution_clock::now();
        copy<<<bd, gd>>>(a,b, init_offset, jstride, kstride, ksize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        t2 = std::chrono::high_resolution_clock::now();
        timings[0] += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        if(!t) {
            for(size_t i=0; i < isize; ++i) {
                for(size_t j=0; j < jsize; ++j) {
                    for(size_t k=0; k < ksize; ++k) {
                        if( b[i+j*jstride + k*kstride + init_offset] != a[i+j*jstride + k*kstride + init_offset] ) {
                            printf("Error in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[i+j*jstride + k*kstride + init_offset], a[i+j*jstride + k*kstride + init_offset]);
                        }
                    }
                }
            }
        }   

        t1 = std::chrono::high_resolution_clock::now();
        sumi1<<<bd, gd>>>(a,b, init_offset, jstride, kstride, ksize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        t2 = std::chrono::high_resolution_clock::now();
        timings[1] += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        if(!t) {
            for(size_t i=0; i < isize; ++i) {
                for(size_t j=0; j < jsize; ++j) {
                    for(size_t k=0; k < ksize; ++k) {
                        if( b[i+j*jstride + k*kstride + init_offset] != a[i+j*jstride + k*kstride + init_offset] + a[i+1 +j*jstride + k*kstride + init_offset] ) {
                            printf("Error in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[i+j*jstride + k*kstride + init_offset], a[i+j*jstride + k*kstride + init_offset]);
                        }
                    }
                }
            }       
        }   
    }


}

int main(int argc, char** argv) {

printf("======================== FLOAT =======================\n");
    launch<float>();
printf("======================== DOUBLE =======================\n");


}
