
template<typename T>
__global__
void FN_NAME(copy) (T* a, T* b, const size_t first_padding, const size_t jstride, const size_t kstride) {
    const unsigned int i = blockIdx.x*gridDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y*gridDim.y + threadIdx.y;

    const unsigned idx = i+j*jstride + first_padding;
    b[idx] = LOAD(a[idx]);
}

template<typename T>
__global__
void FN_NAME(delta) (T* a, T* b, const size_t first_padding, const size_t jstride, const size_t kstride) {
    const unsigned int i = blockIdx.x*gridDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y*gridDim.y + threadIdx.y;

    const unsigned idx = i+j*jstride + first_padding;
    b[idx] = LOAD(a[idx]) ;
}

