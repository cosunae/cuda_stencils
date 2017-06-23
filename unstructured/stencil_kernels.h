#include "cuda_runtime.h"
#include "../tools.h"
#include "../defs.h"
#include "udefs.hpp"
#include "helpers.hpp"
#include "converter.hpp"
#include "umesh.hpp"

template <typename T>
__global__ void copy(T const *__restrict__ a, T *__restrict__ b,
                     const unsigned int init_offset, const unsigned int cstride,
                     const unsigned int jstride, const unsigned int kstride,
                     const unsigned int ksize, const unsigned int isize,
                     const unsigned int jsize) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
                               ? BLOCKSIZEX
                               : isize - blockIdx.x * BLOCKSIZEX;
  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
                               ? BLOCKSIZEY
                               : jsize - blockIdx.y * BLOCKSIZEY;

  unsigned idx = uindex(i, 0, j, cstride, jstride, init_offset);
  for (int k = 0; k < ksize; ++k) {
    if (threadIdx.x < bsi && threadIdx.y < bsj) {
      b[idx] = a[idx];
      b[idx + cstride] = a[idx + cstride];
    }
    idx += kstride;
  }
}

template <typename T>
__global__ void copy_mesh(T const *__restrict__ a, T *__restrict__ b,
                          const unsigned int init_offset,
                          const unsigned int kstride, const unsigned int ksize,
                          const unsigned int mesh_size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < mesh_size) {
    for (int k = 0; k < ksize; ++k) {
      b[idx] = a[idx];
      idx += kstride;
    }
  }
}

template <typename T>
__global__ void on_cells(T const *__restrict__ a, T *__restrict__ b,
                         const unsigned int init_offset,
                         const unsigned int cstride, const unsigned int jstride,
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

  unsigned idx = uindex(i, 0, j, cstride, jstride, init_offset);
  if (threadIdx.x < bsi && threadIdx.y < bsj) {
    for (int k = 0; k < ksize; ++k) {
      // color 0
      b[idx] = a[idx + cstride - 1] + a[idx + cstride] + a[idx - cstride];
      // color 1
      b[idx + cstride] = a[idx] + a[idx + 1] + a[idx + cstride * 2];
      idx += kstride;
    }
  }
}

template <typename T>
__global__ void
on_cells_mesh(T const *__restrict__ a, T *__restrict__ b,
              const unsigned int init_offset, const unsigned int kstride,
              const unsigned int ksize, const unsigned int mesh_size,
              sneighbours_table table) {
  const unsigned int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idx = idx2;

  extern __shared__ size_t tab[];

  const int niter = (table.size() + blockDim.x - 1) / blockDim.x;
  for (int i = 0; i < niter; ++i) {
    const int idx = threadIdx.x * niter * blockDim.x;
    if (idx < table.size()) {
      tab[idx] = table.raw_data(idx);
    }
  }

  const size_t stride =
      table.isize() * table.jsize() * num_colors(table.nloc());

  if (idx < mesh_size) {
    for (int k = 0; k < ksize; ++k) {
      b[idx] = a[idx + tab[idx2 + 0 * stride]] +
               a[idx + tab[idx2 + 1 * stride]] +
               a[idx + tab[idx2 + 2 * stride]];
      idx += kstride;
    }
  }
}

template <typename T>
__global__ void
on_cells_umesh(T const *__restrict__ a, T *__restrict__ b,
               const unsigned int init_offset, const unsigned int kstride,
               const unsigned int ksize, const unsigned int mesh_size,
               uneighbours_table table) {
  const unsigned int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idx = idx2;

  if (idx < mesh_size) {
    for (int k = 0; k < ksize; ++k) {
      b[idx] = a[idx + table(idx2, 0)] + a[idx + table(idx2, 1)] +
               a[idx + table(idx2, 2)];
      idx += kstride;
    }
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
void launch(std::vector<double> &timings, mesh &mesh_, const unsigned int isize,
            const unsigned int jsize, const unsigned ksize,
            const unsigned tsteps, const unsigned warmup_step) {

  const unsigned int halo = 2;
  const unsigned int alignment = 32;
  const unsigned int right_padding = alignment - (isize + halo * 2) % alignment;
  const unsigned int first_padding = alignment - halo;
  const unsigned int cstride_cell = (isize + halo * 2 + right_padding);
  const unsigned int jstride_cell = cstride_cell * num_colors(location::cell);
  const unsigned int kstride_cell = jstride_cell * (jsize + halo * 2);
  const unsigned int total_size_cell =
      first_padding + kstride_cell * (ksize + halo * 2);

  const unsigned int init_offset =
      initial_offset(first_padding, halo, jstride_cell, kstride_cell);

  T *a_cell;
  T *b_cell;
  cudaMallocManaged(&a_cell, sizeof(T) * total_size_cell);
  cudaMallocManaged(&b_cell, sizeof(T) * total_size_cell);

  for (unsigned int i = 0; i < isize; ++i) {
    for (unsigned int c = 0; c < num_colors(location::cell); ++c) {
      for (unsigned int j = 0; j < jsize; ++j) {
        for (unsigned int k = 0; k < ksize; ++k) {
          a_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell, kstride_cell,
                         init_offset)] =
              uindex3(i, c, j, k, cstride_cell, jstride_cell, kstride_cell,
                      init_offset);
        }
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
    copy<<<num_blocks, block_dim>>>(a_cell, b_cell, init_offset, cstride_cell,
                                    jstride_cell, kstride_cell, ksize, isize,
                                    jsize);
    // gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    t2 = std::chrono::high_resolution_clock::now();
    if (t > warmup_step)
      timings[ucopy_st] += std::chrono::duration<double>(t2 - t1).count();
    if (!t) {
      for (unsigned int i = 0; i < isize; ++i) {
        for (unsigned int c = 0; c < num_colors(location::cell); ++c) {
          for (unsigned int j = 0; j < jsize; ++j) {
            for (unsigned int k = 0; k < ksize; ++k) {
              if (b_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                 kstride_cell, init_offset)] !=
                  a_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                 kstride_cell, init_offset)]) {
                printf("Error in (%d,%d,%d) : %f %f\n", (int)i, (int)j, (int)k,
                       b_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                      kstride_cell, init_offset)],
                       a_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                      kstride_cell, init_offset)]);
              }
            }
          }
        }
      }
    }

    //----------------------------------------//
    //--------------  COPY  MESH -------------//
    //----------------------------------------//

    gpuErrchk(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();

    const unsigned int mesh_size =
        mesh_.get_elements(location::cell).last_compute_domain_idx();

    const unsigned int nbx1d =
        (mesh_size + (block_size_x * 2) - 1) / (block_size_x * 2);

    dim3 num_blocks1d(nbx1d, 1, 1);
    dim3 block_dim1d(block_size_x * 2, 1);

    copy_mesh<<<num_blocks1d, block_dim1d>>>(a_cell, b_cell, init_offset,
                                             kstride_cell, ksize, mesh_size);
    // gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    t2 = std::chrono::high_resolution_clock::now();
    if (t > warmup_step)
      timings[ucopymesh_st] += std::chrono::duration<double>(t2 - t1).count();
    if (!t) {
      for (unsigned int i = 0; i < isize; ++i) {
        for (unsigned int c = 0; c < num_colors(location::cell); ++c) {
          for (unsigned int j = 0; j < jsize; ++j) {
            for (unsigned int k = 0; k < ksize; ++k) {
              if (b_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                 kstride_cell, init_offset)] !=
                  a_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                 kstride_cell, init_offset)]) {
                printf("Error in (%d,%d,%d) : %f %f\n", (int)i, (int)j, (int)k,
                       b_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                      kstride_cell, init_offset)],
                       a_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                      kstride_cell, init_offset)]);
              }
            }
          }
        }
      }
    }

    //----------------------------------------//
    //---------------  ONCELLS  --------------//
    //----------------------------------------//

    gpuErrchk(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();
    on_cells<<<num_blocks, block_dim>>>(a_cell, b_cell, init_offset,
                                        cstride_cell, jstride_cell,
                                        kstride_cell, ksize, isize, jsize);
    // gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    t2 = std::chrono::high_resolution_clock::now();
    if (t > warmup_step)
      timings[uoncells_st] += std::chrono::duration<double>(t2 - t1).count();
    if (!t) {
      for (unsigned int i = 0; i < isize; ++i) {
        for (unsigned int j = 0; j < jsize; ++j) {
          for (unsigned int k = 0; k < ksize; ++k) {
            if ((b_cell[uindex3(i, 0, j, k, cstride_cell, jstride_cell,
                                kstride_cell, init_offset)] !=
                 a_cell[uindex3(i - 1, 1, j, k, cstride_cell, jstride_cell,
                                kstride_cell, init_offset)] +
                     a_cell[uindex3(i, 1, j, k, cstride_cell, jstride_cell,
                                    kstride_cell, init_offset)] +
                     a_cell[uindex3(i, 1, j - 1, k, cstride_cell, jstride_cell,
                                    kstride_cell, init_offset)]))
              printf("Error c0 in (%d,%d,%d) : %f %f\n", (int)i, (int)j, (int)k,
                     b_cell[uindex3(i, 0, j, k, cstride_cell, jstride_cell,
                                    kstride_cell, init_offset)],
                     a_cell[uindex3(i, 0, j, k, cstride_cell, jstride_cell,
                                    kstride_cell, init_offset)]);

            if ((b_cell[uindex3(i, 1, j, k, cstride_cell, jstride_cell,
                                kstride_cell, init_offset)] !=
                 a_cell[uindex3(i, 0, j, k, cstride_cell, jstride_cell,
                                kstride_cell, init_offset)] +
                     a_cell[uindex3(i + 1, 0, j, k, cstride_cell, jstride_cell,
                                    kstride_cell, init_offset)] +
                     a_cell[uindex3(i, 0, j + 1, k, cstride_cell, jstride_cell,
                                    kstride_cell, init_offset)])) {
              printf(
                  "Error c1 in (%d,%d,%d) : %f %f\n", (int)i, (int)j, (int)k,
                  b_cell[uindex3(i, 1, j, k, cstride_cell, jstride_cell,
                                 kstride_cell, init_offset)],
                  a_cell[uindex3(i, 0, j, k, cstride_cell, jstride_cell,
                                 kstride_cell, init_offset)] +
                      a_cell[uindex3(i + 1, 0, j, k, cstride_cell, jstride_cell,
                                     kstride_cell, init_offset)] +
                      a_cell[uindex3(i, 0, j + 1, k, cstride_cell, jstride_cell,
                                     kstride_cell, init_offset)]);
            }
          }
        }
      }
    }

    //----------------------------------------//
    //-------------  ONCELLS MESH ------------//
    //----------------------------------------//

    gpuErrchk(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();
    on_cells_mesh<<<num_blocks1d, block_dim1d,
                    mesh_.get_elements(location::cell).table(location::cell) *
                        sizeof(size_t)>>>(
        a_cell, b_cell, init_offset, kstride_cell, ksize, mesh_size,
        mesh_.get_elements(location::cell).table(location::cell));
    // gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    t2 = std::chrono::high_resolution_clock::now();
    if (t > warmup_step)
      timings[uoncellsmesh_st] +=
          std::chrono::duration<double>(t2 - t1).count();
    if (!t) {
      for (unsigned int i = 0; i < isize; ++i) {
        for (unsigned int j = 0; j < jsize; ++j) {
          for (unsigned int k = 0; k < ksize; ++k) {
          }
        }
      }
    }

    //----------------------------------------//
    //--------------  HILBER MESH ------------//
    //----------------------------------------//

    gpuErrchk(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();

    umesh umesh_(mesh_.compd_size(), mesh_.totald_size(),
                 mesh_.nodes_totald_size());

    mesh_to_hilbert(umesh_, mesh_);

    on_cells_umesh<<<num_blocks1d, block_dim1d>>>(
        a_cell, b_cell, init_offset, kstride_cell, ksize, mesh_size,
        umesh_.get_elements(location::cell).table(location::cell));
    gpuErrchk(cudaDeviceSynchronize());

    //    t2 = std::chrono::high_resolution_clock::now();
    //    if (t > warmup_step)
    //      timings[uoncellsmesh_st] +=
    //          std::chrono::duration<double>(t2 - t1).count();
    //    if (!t) {
    //      for (unsigned int i = 0; i < isize; ++i) {
    //        for (unsigned int j = 0; j < jsize; ++j) {
    //          for (unsigned int k = 0; k < ksize; ++k) {
    //          }
    //        }
    //      }
    //    }
  }
}
