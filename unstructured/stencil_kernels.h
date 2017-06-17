#include "cuda_runtime.h"
#include "../tools.h"
#include "../defs.h"

template <typename T>
__global__ void
    FNNAME(copy)(T *__restrict__ a, T *__restrict__ b,
                 const unsigned int init_offset, const unsigned int jstride,
                 const unsigned int kstride, const unsigned int ksize,
                 const unsigned int isize, const unsigned int jsize) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
                               ? BLOCKSIZEX
                               : isize - blockIdx.x * BLOCKSIZEX;
  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
                               ? BLOCKSIZEY
                               : jsize - blockIdx.y * BLOCKSIZEY;

  unsigned idx = index(i, j, jstride, init_offset);
  for (int k = 0; k < ksize; ++k) {
    if (threadIdx.x < bsi && threadIdx.y < bsj) {
      b[idx] = LOAD(a[idx]);
    }
    idx += kstride;
  }
}

// template <typename T>
//__global__ void
//    FNNAME(copyi1)(T *__restrict__ a, T *__restrict__ b,
//                   const unsigned int init_offset, const unsigned int jstride,
//                   const unsigned int kstride, const unsigned int ksize,
//                   const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx + 1]);
//    }
//    idx += kstride;
//  }
//}

// template <typename T>
//__global__ void
//    FNNAME(sumi1)(T *__restrict__ a, T *__restrict__ b,
//                  const unsigned int init_offset, const unsigned int jstride,
//                  const unsigned int kstride, const unsigned int ksize,
//                  const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx]) + LOAD(a[idx + 1]);
//    }
//    idx += kstride;
//  }
//}

// template <typename T>
//__global__ void
//    FNNAME(avgi)(T *__restrict__ a, T *__restrict__ b,
//                 const unsigned int init_offset, const unsigned int jstride,
//                 const unsigned int kstride, const unsigned int ksize,
//                 const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx - 1]) + LOAD(a[idx + 1]);
//    }
//    idx += kstride;
//  }
//}

// template <typename T>
//__global__ void
//    FNNAME(sumj1)(T *__restrict__ a, T *__restrict__ b,
//                  const unsigned int init_offset, const unsigned int jstride,
//                  const unsigned int kstride, const unsigned int ksize,
//                  const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx]) + LOAD(a[idx + jstride]);
//    }
//    idx += kstride;
//  }
//}

// template <typename T>
//__global__ void
//    FNNAME(avgj)(T *__restrict__ a, T *__restrict__ b,
//                 const unsigned int init_offset, const unsigned int jstride,
//                 const unsigned int kstride, const unsigned int ksize,
//                 const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx - jstride]) + LOAD(a[idx + jstride]);
//    }
//    idx += kstride;
//  }
//}

// template <typename T>
//__global__ void
//    FNNAME(sumk1)(T *__restrict__ a, T *__restrict__ b,
//                  const unsigned int init_offset, const unsigned int jstride,
//                  const unsigned int kstride, const unsigned int ksize,
//                  const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx]) + LOAD(a[idx + kstride]);
//    }
//    idx += kstride;
//  }
//}

// template <typename T>
//__global__ void
//    FNNAME(avgk)(T *__restrict__ a, T *__restrict__ b,
//                 const unsigned int init_offset, const unsigned int jstride,
//                 const unsigned int kstride, const unsigned int ksize,
//                 const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx - kstride]) + LOAD(a[idx + kstride]);
//    }
//    idx += kstride;
//  }
//}

// template <typename T>
//__global__ void
//    FNNAME(lap)(T *__restrict__ a, T *__restrict__ b,
//                const unsigned int init_offset, const unsigned int jstride,
//                const unsigned int kstride, const unsigned int ksize,
//                const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx]) + LOAD(a[idx + 1]) + LOAD(a[idx - 1]) +
//               LOAD(a[idx + jstride]) + LOAD(a[idx - jstride]);
//    }
//    idx += kstride;
//  }
//}

template <typename T>
void FNNAME(launch)(std::vector<double> &timings, const unsigned int isize,
                    const unsigned int jsize, const unsigned ksize,
                    const unsigned tsteps, const unsigned warmup_step) {

  const unsigned int halo = 2;
  const unsigned int alignment = 32;
  const unsigned int right_padding = alignment - (isize + halo * 2) % alignment;
  const unsigned int first_padding = alignment - halo;
  const unsigned int jstride = (isize + halo * 2 + right_padding);
  const unsigned int kstride = jstride * (jsize + halo * 2);
  const unsigned int total_size = first_padding + kstride * (ksize + halo * 2);

  const unsigned int init_offset =
      initial_offset(first_padding, halo, jstride, kstride);

  T *a;
  T *b;
  cudaMallocManaged(&a, sizeof(T) * total_size);
  cudaMallocManaged(&b, sizeof(T) * total_size);

  for (unsigned int i = 0; i < isize; ++i) {
    for (unsigned int j = 0; j < jsize; ++j) {
      for (unsigned int k = 0; k < ksize; ++k) {
        a[i + j * jstride + k * kstride + init_offset] =
            i + j * jstride + k * kstride + init_offset;
      }
    }
  }

  const unsigned int block_size_x = BLOCKSIZEX;
  const unsigned int block_size_y = BLOCKSIZEY;

  const unsigned int nbx = (isize + block_size_x - 1) / block_size_x;
  const unsigned int nby = (jsize + block_size_y - 1) / block_size_y;

  dim3 num_blocks(nbx, nby, 1);
  dim3 block_dim(block_size_x, block_size_y);

  std::chrono::high_resolution_clock::time_point t1, t2;

  for (unsigned int t = 0; t < tsteps; t++) {

    //----------------------------------------//
    //----------------  COPY  ----------------//
    //----------------------------------------//

    gpuErrchk(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();
    FNNAME(copy)<<<num_blocks, block_dim>>>(a, b, init_offset, jstride, kstride,
                                            ksize, isize, jsize);
    // gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    t2 = std::chrono::high_resolution_clock::now();
    if (t > warmup_step)
      timings[copy_st] += std::chrono::duration<double>(t2 - t1).count();
    if (!t) {
      for (unsigned int i = 0; i < isize; ++i) {
        for (unsigned int j = 0; j < jsize; ++j) {
          for (unsigned int k = 0; k < ksize; ++k) {
            if (b[i + j * jstride + k * kstride + init_offset] !=
                a[i + j * jstride + k * kstride + init_offset]) {
              printf("Error in (%d,%d,%d) : %f %f\n", (int)i, (int)j, (int)k,
                     b[i + j * jstride + k * kstride + init_offset],
                     a[i + j * jstride + k * kstride + init_offset]);
            }
          }
        }
      }
    }

    //    //----------------------------------------//
    //    //---------------- COPYi1 ----------------//
    //    //----------------------------------------//
    //    gpuErrchk(cudaDeviceSynchronize());
    //    t1 = std::chrono::high_resolution_clock::now();
    //    FNNAME(copyi1)<<<num_blocks, block_dim>>>(a, b, init_offset, jstride,
    //                                              kstride, ksize, isize,
    //                                              jsize);
    //    // gpuErrchk(cudaPeekAtLastError());
    //    gpuErrchk(cudaDeviceSynchronize());

    //    t2 = std::chrono::high_resolution_clock::now();
    //    if (t > warmup_step)
    //      timings[copyi1_st] += std::chrono::duration<double>(t2 -
    //      t1).count();
    //    if (!t) {
    //      for (unsigned int i = 0; i < isize; ++i) {
    //        for (unsigned int j = 0; j < jsize; ++j) {
    //          for (unsigned int k = 0; k < ksize; ++k) {
    //            if (b[i + j * jstride + k * kstride + init_offset] !=
    //                a[i + 1 + j * jstride + k * kstride + init_offset]) {
    //              printf("Error in (%d,%d,%d) : %f %f\n", (int)i, (int)j,
    //              (int)k,
    //                     b[i + j * jstride + k * kstride + init_offset],
    //                     a[i + j * jstride + k * kstride + init_offset]);
    //            }
    //          }
    //        }
    //      }
    //    }

    //    //----------------------------------------//
    //    //----------------  SUMi1 ----------------//
    //    //----------------------------------------//

    //    gpuErrchk(cudaDeviceSynchronize());
    //    t1 = std::chrono::high_resolution_clock::now();
    //    FNNAME(sumi1)<<<num_blocks, block_dim>>>(a, b, init_offset, jstride,
    //                                             kstride, ksize, isize,
    //                                             jsize);
    //    // gpuErrchk(cudaPeekAtLastError());
    //    gpuErrchk(cudaDeviceSynchronize());

    //    t2 = std::chrono::high_resolution_clock::now();
    //    if (t > warmup_step)
    //      timings[sumi1_st] +=
    //          std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
    //          t1)
    //              .count();

    //    if (!t) {
    //      for (unsigned int i = 0; i < isize; ++i) {
    //        for (unsigned int j = 0; j < jsize; ++j) {
    //          for (unsigned int k = 0; k < ksize; ++k) {
    //            if (b[i + j * jstride + k * kstride + init_offset] !=
    //                a[i + j * jstride + k * kstride + init_offset] +
    //                    a[i + 1 + j * jstride + k * kstride + init_offset]) {
    //              printf("Error in (%d,%d,%d) : %f %f\n", (int)i, (int)j,
    //              (int)k,
    //                     b[i + j * jstride + k * kstride + init_offset],
    //                     a[i + j * jstride + k * kstride + init_offset]);
    //            }
    //          }
    //        }
    //      }
    //    }

    //    //----------------------------------------//
    //    //----------------  SUMj1 ----------------//
    //    //----------------------------------------//

    //    gpuErrchk(cudaDeviceSynchronize());
    //    t1 = std::chrono::high_resolution_clock::now();
    //    FNNAME(sumj1)<<<num_blocks, block_dim>>>(a, b, init_offset, jstride,
    //                                             kstride, ksize, isize,
    //                                             jsize);
    //    // gpuErrchk(cudaPeekAtLastError());
    //    gpuErrchk(cudaDeviceSynchronize());

    //    t2 = std::chrono::high_resolution_clock::now();
    //    if (t > warmup_step)
    //      timings[sumj1_st] +=
    //          std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
    //          t1)
    //              .count();

    //    if (!t) {
    //      for (unsigned int i = 0; i < isize; ++i) {
    //        for (unsigned int j = 0; j < jsize; ++j) {
    //          for (unsigned int k = 0; k < ksize; ++k) {
    //            if (b[i + j * jstride + k * kstride + init_offset] !=
    //                a[i + j * jstride + k * kstride + init_offset] +
    //                    a[i + (j + 1) * jstride + k * kstride + init_offset])
    //                    {
    //              printf("Error in (%d,%d,%d) : %f %f\n", (int)i, (int)j,
    //              (int)k,
    //                     b[i + j * jstride + k * kstride + init_offset],
    //                     a[i + j * jstride + k * kstride + init_offset]);
    //            }
    //          }
    //        }
    //      }
    //    }

    //    //----------------------------------------//
    //    //----------------  SUMk1 ----------------//
    //    //----------------------------------------//

    //    gpuErrchk(cudaDeviceSynchronize());
    //    t1 = std::chrono::high_resolution_clock::now();
    //    FNNAME(sumk1)<<<num_blocks, block_dim>>>(a, b, init_offset, jstride,
    //                                             kstride, ksize, isize,
    //                                             jsize);
    //    // gpuErrchk(cudaPeekAtLastError());
    //    gpuErrchk(cudaDeviceSynchronize());

    //    t2 = std::chrono::high_resolution_clock::now();
    //    if (t > warmup_step)
    //      timings[sumk1_st] +=
    //          std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
    //          t1)
    //              .count();

    //    if (!t) {
    //      for (unsigned int i = 0; i < isize; ++i) {
    //        for (unsigned int j = 0; j < jsize; ++j) {
    //          for (unsigned int k = 0; k < ksize; ++k) {
    //            if (b[i + j * jstride + k * kstride + init_offset] !=
    //                a[i + j * jstride + k * kstride + init_offset] +
    //                    a[i + j * jstride + (k + 1) * kstride + init_offset])
    //                    {
    //              printf("Error in (%d,%d,%d) : %f %f\n", (int)i, (int)j,
    //              (int)k,
    //                     b[i + j * jstride + k * kstride + init_offset],
    //                     a[i + j * jstride + k * kstride + init_offset]);
    //            }
    //          }
    //        }
    //      }
    //    }

    //    //----------------------------------------//
    //    //----------------  AVGi  ----------------//
    //    //----------------------------------------//

    //    gpuErrchk(cudaDeviceSynchronize());
    //    t1 = std::chrono::high_resolution_clock::now();
    //    FNNAME(avgi)<<<num_blocks, block_dim>>>(a, b, init_offset, jstride,
    //    kstride,
    //                                            ksize, isize, jsize);
    //    // gpuErrchk(cudaPeekAtLastError());
    //    gpuErrchk(cudaDeviceSynchronize());

    //    t2 = std::chrono::high_resolution_clock::now();
    //    if (t > warmup_step)
    //      timings[avgi_st] +=
    //          std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
    //          t1)
    //              .count();

    //    if (!t) {
    //      for (unsigned int i = 0; i < isize; ++i) {
    //        for (unsigned int j = 0; j < jsize; ++j) {
    //          for (unsigned int k = 0; k < ksize; ++k) {
    //            if (b[i + j * jstride + k * kstride + init_offset] !=
    //                a[i - 1 + j * jstride + k * kstride + init_offset] +
    //                    a[i + 1 + j * jstride + k * kstride + init_offset]) {
    //              printf("Error in (%d,%d,%d) : %f %f\n", (int)i, (int)j,
    //              (int)k,
    //                     b[i + j * jstride + k * kstride + init_offset],
    //                     a[i + j * jstride + k * kstride + init_offset]);
    //            }
    //          }
    //        }
    //      }
    //    }

    //    //----------------------------------------//
    //    //----------------  AVGj  ----------------//
    //    //----------------------------------------//

    //    gpuErrchk(cudaDeviceSynchronize());
    //    t1 = std::chrono::high_resolution_clock::now();
    //    FNNAME(avgj)<<<num_blocks, block_dim>>>(a, b, init_offset, jstride,
    //    kstride,
    //                                            ksize, isize, jsize);
    //    // gpuErrchk(cudaPeekAtLastError());
    //    gpuErrchk(cudaDeviceSynchronize());

    //    t2 = std::chrono::high_resolution_clock::now();
    //    if (t > warmup_step)
    //      timings[avgj_st] +=
    //          std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
    //          t1)
    //              .count();

    //    if (!t) {
    //      for (unsigned int i = 0; i < isize; ++i) {
    //        for (unsigned int j = 0; j < jsize; ++j) {
    //          for (unsigned int k = 0; k < ksize; ++k) {
    //            if (b[i + j * jstride + k * kstride + init_offset] !=
    //                a[i + (j - 1) * jstride + k * kstride + init_offset] +
    //                    a[i + (j + 1) * jstride + k * kstride + init_offset])
    //                    {
    //              printf("Error in (%d,%d,%d) : %f %f\n", (int)i, (int)j,
    //              (int)k,
    //                     b[i + j * jstride + k * kstride + init_offset],
    //                     a[i + j * jstride + k * kstride + init_offset]);
    //            }
    //          }
    //        }
    //      }
    //    }

    //    //----------------------------------------//
    //    //----------------  AVGk  ----------------//
    //    //----------------------------------------//

    //    gpuErrchk(cudaDeviceSynchronize());
    //    t1 = std::chrono::high_resolution_clock::now();
    //    FNNAME(avgk)<<<num_blocks, block_dim>>>(a, b, init_offset, jstride,
    //    kstride,
    //                                            ksize, isize, jsize);
    //    // gpuErrchk(cudaPeekAtLastError());
    //    gpuErrchk(cudaDeviceSynchronize());

    //    t2 = std::chrono::high_resolution_clock::now();
    //    if (t > warmup_step)
    //      timings[avgk_st] +=
    //          std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
    //          t1)
    //              .count();

    //    if (!t) {
    //      for (unsigned int i = 0; i < isize; ++i) {
    //        for (unsigned int j = 0; j < jsize; ++j) {
    //          for (unsigned int k = 0; k < ksize; ++k) {
    //            if (b[i + j * jstride + k * kstride + init_offset] !=
    //                a[i + j * jstride + (k - 1) * kstride + init_offset] +
    //                    a[i + j * jstride + (k + 1) * kstride + init_offset])
    //                    {
    //              printf("Error in (%d,%d,%d) : %f %f\n", (int)i, (int)j,
    //              (int)k,
    //                     b[i + j * jstride + k * kstride + init_offset],
    //                     a[i + j * jstride + k * kstride + init_offset]);
    //            }
    //          }
    //        }
    //      }
    //    }

    //    //----------------------------------------//
    //    //----------------  LAP   ----------------//
    //    //----------------------------------------//

    //    gpuErrchk(cudaDeviceSynchronize());
    //    t1 = std::chrono::high_resolution_clock::now();
    //    FNNAME(lap)<<<num_blocks, block_dim>>>(a, b, init_offset, jstride,
    //    kstride,
    //                                           ksize, isize, jsize);
    //    // gpuErrchk(cudaPeekAtLastError());
    //    gpuErrchk(cudaDeviceSynchronize());

    //    t2 = std::chrono::high_resolution_clock::now();
    //    if (t > warmup_step)
    //      timings[lap_st] +=
    //          std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
    //          t1)
    //              .count();

    //    if (!t) {
    //      for (unsigned int i = 0; i < isize; ++i) {
    //        for (unsigned int j = 0; j < jsize; ++j) {
    //          for (unsigned int k = 0; k < ksize; ++k) {
    //            if (b[i + j * jstride + k * kstride + init_offset] !=
    //                a[i + j * jstride + k * kstride + init_offset] +
    //                    a[i + 1 + j * jstride + k * kstride + init_offset] +
    //                    a[i - 1 + j * jstride + k * kstride + init_offset] +
    //                    a[i + (j + 1) * jstride + k * kstride + init_offset] +
    //                    a[i + (j - 1) * jstride + k * kstride + init_offset])
    //                    {
    //              printf("Error in (%d,%d,%d) : %f %f\n", (int)i, (int)j,
    //              (int)k,
    //                     b[i + j * jstride + k * kstride + init_offset],
    //                     a[i + j * jstride + k * kstride + init_offset]);
    //            }
    //          }
    //        }
    //      }
    //    }
  }
}
