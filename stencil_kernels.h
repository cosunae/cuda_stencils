#include "tools.h"
#include "defs.h"

template<typename T>
__global__
void FNNAME(copy) ( T* __restrict__ a,  T* __restrict__ b, const size_t init_offset, const size_t jstride, const size_t kstride, const size_t ksize) {
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned idx = index(i,j,jstride, init_offset);
    for(int k=0; k < ksize; ++k) {
if(threadIdx.x==0 && ((uintptr_t)(&b[idx]))%32 != 0 && ((uintptr_t)(&a[idx]))%32 != 0 ) printf("ERROR");
        b[idx] = LOAD(a[idx]);
        idx += kstride;
    }
}

template<typename T>
__global__
void FNNAME(copyi1) ( T  *  __restrict__ a,  T  * __restrict__ b, const size_t init_offset, const size_t jstride, const size_t kstride, const size_t ksize) {
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned idx = index(i,j,jstride, init_offset);
    for(int k=0; k < ksize; ++k) {
        b[idx] = LOAD(a[idx+1]);
        idx += kstride;
    }
}

template<typename T>
__global__
void FNNAME(sumi1) ( T  * __restrict__ a,  T  * __restrict__ b, const size_t init_offset, const size_t jstride, const size_t kstride, const size_t ksize) {
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;


    unsigned idx = index(i,j,jstride, init_offset);
    for(int k=0; k < ksize; ++k) {
        b[idx] = LOAD(a[idx]) + LOAD(a[idx+1]) ;
        idx += kstride;
    }
}

template<typename T>
__global__
void FNNAME(sumj1) ( T  * __restrict__ a,  T  * __restrict__ b, const size_t init_offset, const size_t jstride, const size_t kstride, const size_t ksize) {
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned idx = index(i,j,jstride, init_offset);
    for(int k=0; k < ksize; ++k) {
        b[idx] = LOAD(a[idx]) + LOAD(a[idx+jstride]) ;
        idx += kstride;
    }
}

template<typename T>
__global__
void FNNAME(sumk1) ( T  * __restrict__ a,  T  * __restrict__ b, const size_t init_offset, const size_t jstride, const size_t kstride, const size_t ksize) {
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned idx = index(i,j,jstride, init_offset);
    for(int k=0; k < ksize; ++k) {
        b[idx] = LOAD(a[idx]) + LOAD(a[idx+kstride]) ;
        idx += kstride;
    }
}

template<typename T>
__global__
void FNNAME(lap) ( T  * __restrict__ a,  T  * __restrict__ b, const size_t init_offset, const size_t jstride, const size_t kstride, const size_t ksize) {
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned idx = index(i,j,jstride, init_offset);
    for(int k=0; k < ksize; ++k) {
        b[idx] = LOAD(a[idx]) + LOAD(a[idx+1]) + LOAD(a[idx-1]) + LOAD(a[idx+jstride]) + LOAD(a[idx-jstride]);
        idx += kstride;
    }
}


template<typename T>
void FNNAME(launch) ( std::vector<double>& timings, const unsigned int isize, const unsigned int jsize, const unsigned ksize, const unsigned tsteps, const unsigned warmup_step ) {

    const size_t halo=2;
    const size_t alignment=32;
    const size_t right_padding=(isize+halo*2)%alignment;
    const size_t first_padding=alignment-halo;
    const size_t total_size=first_padding+(isize+halo*2+right_padding)*(jsize+halo*2)*(ksize+halo*2);
    const size_t jstride = (isize+halo*2+right_padding);
    const size_t kstride = jstride*(jsize+halo*2);

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

    std::chrono::high_resolution_clock::time_point t1,t2;

    timings[copy_st] = 0;

    for(size_t t=0; t < tsteps; t++) {

        /******************************************/
        /**************** COPY   ******************/
        /******************************************/
        gpuErrchk(cudaDeviceSynchronize());
        t1 = std::chrono::high_resolution_clock::now();
        FNNAME(copy)<<<bd, gd>>>(a,b, init_offset, jstride, kstride, ksize);
        //gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[copy_st] += std::chrono::duration<double>(t2-t1).count();
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

        /******************************************/
        /***************** COPYi1 *****************/
        /******************************************/
        gpuErrchk(cudaDeviceSynchronize());
        t1 = std::chrono::high_resolution_clock::now();
        FNNAME(copyi1)<<<bd, gd>>>(a,b, init_offset, jstride, kstride, ksize);
        //gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[copyi1_st] += std::chrono::duration<double>(t2-t1).count();
        if(!t) {
            for(size_t i=0; i < isize; ++i) {
                for(size_t j=0; j < jsize; ++j) {
                    for(size_t k=0; k < ksize; ++k) {
                        if( b[i+j*jstride + k*kstride + init_offset] != a[i+1+j*jstride + k*kstride + init_offset] ) {
                            printf("Error in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[i+j*jstride + k*kstride + init_offset], a[i+j*jstride + k*kstride + init_offset]);
                        }
                    }
                }
            }
        }

        /******************************************/
        /***************** SUMi1 *****************/
        /******************************************/
 
        gpuErrchk(cudaDeviceSynchronize());
        t1 = std::chrono::high_resolution_clock::now();
        FNNAME(sumi1)<<<bd, gd>>>(a,b, init_offset, jstride, kstride, ksize);
        //gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[sumi1_st] += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

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


        /******************************************/
        /***************** SUMj1 *****************/
        /******************************************/
 
        gpuErrchk(cudaDeviceSynchronize());
        t1 = std::chrono::high_resolution_clock::now();
        FNNAME(sumj1)<<<bd, gd>>>(a,b, init_offset, jstride, kstride, ksize);
        //gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[sumj1_st] += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        if(!t) {
            for(size_t i=0; i < isize; ++i) {
                for(size_t j=0; j < jsize; ++j) {
                    for(size_t k=0; k < ksize; ++k) {
                        if( b[i+j*jstride + k*kstride + init_offset] != a[i+j*jstride + k*kstride + init_offset] + a[i + (j+1)*jstride + k*kstride + init_offset] ) {
                            printf("Error in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[i+j*jstride + k*kstride + init_offset], a[i+j*jstride + k*kstride + init_offset]);
                        }
                    }
                }
            }
        }
 
        /******************************************/
        /***************** SUMk1 *****************/
        /******************************************/
 
        gpuErrchk(cudaDeviceSynchronize());
        t1 = std::chrono::high_resolution_clock::now();
        FNNAME(sumk1)<<<bd, gd>>>(a,b, init_offset, jstride, kstride, ksize);
        //gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[sumk1_st] += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        if(!t) {
            for(size_t i=0; i < isize; ++i) {
                for(size_t j=0; j < jsize; ++j) {
                    for(size_t k=0; k < ksize; ++k) {
                        if( b[i+j*jstride + k*kstride + init_offset] != a[i+j*jstride + k*kstride + init_offset] + a[i +j*jstride + (k+1)*kstride + init_offset] ) {
                            printf("Error in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[i+j*jstride + k*kstride + init_offset], a[i+j*jstride + k*kstride + init_offset]);
                        }
                    }
                }
            }
        }
 
        /******************************************/
        /*****************  LAP   *****************/
        /******************************************/
 
        gpuErrchk(cudaDeviceSynchronize());
        t1 = std::chrono::high_resolution_clock::now();
        FNNAME(lap)<<<bd, gd>>>(a,b, init_offset, jstride, kstride, ksize);
        //gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[lap_st] += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        if(!t) {
            for(size_t i=0; i < isize; ++i) {
                for(size_t j=0; j < jsize; ++j) {
                    for(size_t k=0; k < ksize; ++k) {
                        if( b[i+j*jstride + k*kstride + init_offset] != a[i+j*jstride + k*kstride + init_offset] + a[i+1 +j*jstride + k*kstride + init_offset] +
                               a[i-1 +j*jstride + k*kstride + init_offset] + a[i+(j+1)*jstride + k*kstride + init_offset] + a[i+(j-1)*jstride + k*kstride + init_offset] ) {
                            printf("Error in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[i+j*jstride + k*kstride + init_offset], a[i+j*jstride + k*kstride + init_offset]);
                        }
                    }
                }
            }
        }
 
    }

}

