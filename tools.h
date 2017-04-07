#pragma once

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ __device__
inline unsigned int initial_offset(const unsigned int first_padding, const unsigned int halo, const unsigned jstride, const unsigned kstride) {
    return first_padding + halo + halo*jstride + halo*kstride;
}

__host__ __device__
inline unsigned int index(const unsigned int i, const unsigned int j, const unsigned jstride, const unsigned first_padding) {
    return i+j*jstride + first_padding;
}

