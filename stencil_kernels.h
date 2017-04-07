#include "tools.h"

template<typename T>
__global__
void FNNAME(copy) (T* a, T* b, const size_t init_offset, const size_t jstride, const size_t kstride, const size_t ksize) {
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned idx = index(i,j,jstride, init_offset);
    for(int k=0; k < ksize; ++k) {
        b[idx] = LOAD(a[idx]);
        idx += kstride;
    }
}

template<typename T>
__global__
void FNNAME(copyi1) (T* a, T* b, const size_t init_offset, const size_t jstride, const size_t kstride, const size_t ksize) {
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
void FNNAME(sumi1) (T* a, T* b, const size_t init_offset, const size_t jstride, const size_t kstride, const size_t ksize) {
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
void FNNAME(sumj1) (T* a, T* b, const size_t init_offset, const size_t jstride, const size_t kstride, const size_t ksize) {
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
void FNNAME(sumk1) (T* a, T* b, const size_t init_offset, const size_t jstride, const size_t kstride, const size_t ksize) {
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
void FNNAME(lap) (T* a, T* b, const size_t init_offset, const size_t jstride, const size_t kstride, const size_t ksize) {
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned idx = index(i,j,jstride, init_offset);
    for(int k=0; k < ksize; ++k) {
        b[idx] = LOAD(a[idx]) + LOAD(a[idx+1]) + LOAD(a[idx-1]) + LOAD(a[idx+jstride]) + LOAD(a[idx-jstride]);
        idx += kstride;
    }
}


