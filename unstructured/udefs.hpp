#pragma once

enum ustencils { ucopy_st = 0, ucopymesh_st, uoncells_st, uoncellsmesh_st, unum_bench_st };

enum class location { cell = 0, edge, vertex };

__host__ __device__
constexpr unsigned int num_colors(location loc) {
  return loc == location::cell ? 2 : (loc == location::edge ? 3 : 1);
}

__host__ __device__
static constexpr size_t num_nodes(const unsigned int isize,
                                  const unsigned int jsize,
                                  const unsigned int nhalo) {
  return (isize + nhalo * 2) * (jsize + nhalo * 2);
}

__host__ __device__
static constexpr size_t num_neighbours(const location primary_loc,
                                       const location neigh_loc) {
  return primary_loc == location::cell
             ? 3
             : (primary_loc == location::edge
                    ? (neigh_loc == location::edge ? 4 : 2)
                    : (6));
}
