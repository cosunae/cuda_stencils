#pragma once

#define BLOCKSIZEX 32
#define BLOCKSIZEY 8

enum stencils {
  copy_st = 0,
  copyi1_st,
  sumi1_st,
  sumj1_st,
  sumk1_st,
  avgi_st,
  avgj_st,
  avgk_st,
  lap_st,
  num_bench_st
};

#ifdef __GNUC__
#define GT_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define GT_FORCE_INLINE inline __forceinline
#else
#define GT_FORCE_INLINE inline
#endif

#ifndef GT_FUNCTION
#ifdef __CUDACC__
#define GT_FUNCTION __host__ __device__ __forceinline__
#define GT_FUNCTION_HOST __host__ __forceinline__
#define GT_FUNCTION_DEVICE __device__ __forceinline__
#define GT_FUNCTION_WARNING __host__ __device__
#else
#define GT_FUNCTION GT_FORCE_INLINE
#define GT_FUNCTION_HOST GT_FORCE_INLINE
#define GT_FUNCTION_DEVICE GT_FORCE_INLINE
#define GT_FUNCTION_WARNING
#endif
#endif
