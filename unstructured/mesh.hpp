/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once
#include "array.hpp"
#include <sstream>
#include <iostream>
#include <fstream>
#include <array>

#define GHOST_ID_X 10000
#define GHOST_ID_Y 20000

enum class location { cell = 0, edge, vertex };

constexpr unsigned int num_colors(location loc) {
  return loc == location::cell ? 2 : (loc == location::edge ? 3 : 1);
}

static constexpr size_t num_nodes(const unsigned int isize,
                                  const unsigned int jsize,
                                  const unsigned int nhalo) {
  return (isize + nhalo * 2) * (jsize + nhalo * 2);
}

static constexpr size_t num_neighbours(const location primary_loc,
                                       const location neigh_loc) {
  return primary_loc == location::cell
             ? 3
             : (primary_loc == location::edge
                    ? (neigh_loc == location::edge ? 4 : 2)
                    : (6));
}

class neighbours_table {
public:
  static constexpr size_t size_of_array(const location primary_loc,
                                        const location neigh_loc,
                                        const unsigned int isize,
                                        const unsigned int jsize,
                                        const unsigned int nhalo) {
    return num_nodes(isize, jsize, nhalo) * num_colors(primary_loc) *
           sizeof(size_t) * num_neighbours(primary_loc, neigh_loc);
  }

  neighbours_table(location primary_loc, location neigh_loc, size_t isize,
                   size_t jsize, size_t nhalo)
      : m_ploc(primary_loc), m_nloc(neigh_loc), m_isize(isize), m_jsize(jsize),
        m_nhalo(nhalo) {
#ifdef ENABLE_GPU
    cudaMallocManaged(
        &m_data, size_of_array(primary_loc, neigh_loc, isize, jsize, nhalo));
#else
    m_data = (size_t *)malloc(
        size_of_array(primary_loc, neigh_loc, isize, jsize, nhalo));
#endif
  }

  size_t &operator()(int i, unsigned int c, int j, unsigned int neigh_idx) {

    assert(index_in_tables(i, c, j, neigh_idx) <
           size_of_array(m_ploc, m_nloc, m_isize, m_jsize, m_nhalo));

    return m_data[index_in_tables(i, c, j, neigh_idx)];
  }

  size_t index_in_tables(int i, unsigned int c, int j, unsigned int neigh_idx) {
    assert(i > -(int)m_nhalo && i < (int)(m_isize + m_nhalo));
    assert(c < num_colors(m_ploc));
    assert(j > -(int)m_nhalo && j < (int)(m_jsize + m_nhalo));
    assert(neigh_idx < num_neighbours(m_ploc, m_nloc));
    return (i + (int)m_nhalo) + c * (m_isize + m_nhalo * 2) +
           (j + (int)m_nhalo) * (m_isize + m_nhalo * 2) * num_colors(m_ploc) +
           neigh_idx * (m_isize + m_nhalo * 2) * num_colors(m_ploc) *
               (m_jsize + m_nhalo * 2);
  }

private:
  location m_ploc;
  location m_nloc;
  size_t m_isize, m_jsize, m_nhalo;
  size_t *m_data;
};

class elements {
  static constexpr size_t size_of_array(const location primary_loc,
                                        const unsigned int isize,
                                        const unsigned int jsize,
                                        const unsigned int nhalo) {
    return num_nodes(isize, jsize, nhalo) * num_colors(primary_loc) *
           sizeof(size_t);
  }

  elements(location primary_loc, size_t isize, size_t jsize, size_t nhalo)
      : m_loc(primary_loc), m_isize(isize), m_jsize(jsize), m_nhalo(nhalo),
        m_elements_to_cells(primary_loc, location::cell, isize, jsize, nhalo),
        m_elements_to_edges(primary_loc, location::edge, isize, jsize, nhalo),
        m_elements_to_vertices(primary_loc, location::vertex, isize, jsize,
                               nhalo) {
#ifdef ENABLE_GPU
    cudaMallocManaged(&m_idx, size_of_array(primary_loc, isize, jsize, nhalo));
#else
    m_idx = (size_t *)malloc(size_of_array(primary_loc, isize, jsize, nhalo));
#endif
  }

  size_t &neighbor(location neigh_loc, int i, unsigned int c, int j,
                   unsigned int neigh_idx) {
    if (neigh_loc == location::cell)
      return m_elements_to_cells(i, c, j, neigh_idx);
    else if (neigh_loc == location::edge)
      return m_elements_to_cells(i, c, j, neigh_idx);
    return m_elements_to_vertices(i, c, j, neigh_idx);
  }

  size_t last_compute_domain_idx() { return m_isize * m_jsize; }

  virtual size_t idx(int i, unsigned int c, int j) {
    if (i >= 0 && i < (int)m_isize && j >= 0 && j < (int)m_jsize)
      return i + c * m_isize + j * m_isize * num_colors(m_loc);
    if (i < 0 && j >= 0 && j < (int)m_jsize)
      return last_compute_domain_idx() - i + c * m_nhalo +
             j * m_nhalo * num_colors(m_loc);
    if (j < 0 && i < (int)m_isize)
      return last_compute_domain_idx() + m_nhalo * num_colors(m_loc) * m_jsize +
             (j + (int)m_nhalo) + c * m_nhalo;
    // TODO
    return -1;
  }

private:
  location m_loc;
  size_t *m_idx;
  size_t m_isize, m_jsize, m_nhalo;
  neighbours_table m_elements_to_cells, m_elements_to_edges,
      m_elements_to_vertices;
};

class nodes {

  template <typename T>
  static constexpr size_t
  size_of_mesh_fields(const location loc, const unsigned int isize,
                      const unsigned int jsize, const unsigned int nhalo) {
    return num_nodes(isize, jsize, nhalo) * num_colors(loc) * sizeof(T);
  }

  nodes(size_t isize, size_t jsize, size_t nhalo)
      : m_isize(isize), m_jsize(jsize), m_nhalo(nhalo),
        m_vertex_to_cell(location::vertex, location::cell, isize, jsize, nhalo),
        m_vertex_to_edge(location::vertex, location::edge, isize, jsize, nhalo),
        m_vertex_to_vertex(location::vertex, location::vertex, isize, jsize,
                           nhalo) {
#ifdef ENABLE_GPU
    cudaMallocManaged(&m_x, size_of_mesh_fields<double>(location::vertex, isize,
                                                        jsize, nhalo));
    cudaMallocManaged(&m_y, size_of_mesh_fields<double>(location::vertex, isize,
                                                        jsize, nhalo));
#else
    m_x = (double *)malloc(
        size_of_mesh_fields<double>(location::vertex, isize, jsize, nhalo));
    m_y = (double *)malloc(
        size_of_mesh_fields<double>(location::vertex, isize, jsize, nhalo));
#endif
  }
  size_t last_compute_domain_idx() { return m_isize * m_jsize; }

  virtual size_t idx(int i, unsigned int c, int j) {
    if (i >= 0 && i < (int)m_isize && j >= 0 && j < (int)m_jsize)
      return i + c * m_isize + j * m_isize * num_colors(location::vertex);
    if (i == (int)m_isize && j >= 0 && j < (int)m_jsize)
      return last_compute_domain_idx() + j;
    if (j == (int)m_jsize && i >= 0)
      return last_compute_domain_idx() + m_jsize + i;
    if (i < 0 && j >= 0 && j < (int)m_jsize)
      return last_compute_domain_idx() + m_jsize + m_isize + 1 - i +
             c * m_nhalo + j * m_nhalo * num_colors(location::vertex);
    if (j < 0 && i < (int)m_isize)
      return last_compute_domain_idx() + m_jsize + m_isize + 1 +
             m_nhalo * num_colors(location::vertex) * m_jsize +
             (j + (int)m_nhalo) + c * m_nhalo;
    // TODO
    return -1;
  }
  double &x(unsigned int idx) { return m_x[idx]; }
  double &y(unsigned int idx) { return m_y[idx]; }

private:
  size_t m_isize, m_jsize, m_nhalo;
  double *m_x;
  double *m_y;
  neighbours_table m_vertex_to_cell;
  neighbours_table m_vertex_to_edge;
  neighbours_table m_vertex_to_vertex;
};

//////////////// conventions ///////////////////
/// cell to vertex
/// 0 -------- 1
///   \      /
///    \    /
///     \  /
///      2

///
///      0
///     / \
///    /   \
///   /     \
///  2-------1

class mesh {

  size_t end_idx_compute_domain() { return m_isize * m_jsize; }

  template <typename T>
  static constexpr size_t
  size_of_mesh_fields(const location loc, const unsigned int isize,
                      const unsigned int jsize, const unsigned int nhalo) {
    return num_nodes(isize, jsize, nhalo) * num_colors(loc) * sizeof(T);
  }

public:
  mesh(const unsigned int isize, const unsigned int jsize,
       const unsigned int nhalo)
      : m_isize(isize), m_jsize(jsize), m_nhalo(nhalo), m_i_domain{0, isize},
        m_j_domain{0, jsize},
        m_cell_to_cells(location::cell, location::cell, isize, jsize, nhalo),
        m_cell_to_edges(location::cell, location::edge, isize, jsize, nhalo),
        m_cell_to_vertices(location::cell, location::vertex, isize, jsize,
                           nhalo),
        m_edge_to_cells(location::edge, location::cell, isize, jsize, nhalo),
        m_edge_to_edges(location::edge, location::edge, isize, jsize, nhalo),
        m_edge_to_vertices(location::edge, location::vertex, isize, jsize,
                           nhalo),
        m_vertex_to_cells(location::vertex, location::cell, isize, jsize,
                          nhalo),
        m_vertex_to_edges(location::vertex, location::edge, isize, jsize,
                          nhalo),
        m_vertex_to_vertices(location::vertex, location::vertex, isize, jsize,
                             nhalo) {
#ifdef ENABLE_GPU
    cudaMallocManaged(&m_x, size_of_mesh_fields<double>(location::vertex, isize,
                                                        jsize, nhalo));
    cudaMallocManaged(&m_y, size_of_mesh_fields<double>(location::vertex, isize,
                                                        jsize, nhalo));
#else
    m_x = (double *)malloc(
        size_of_mesh_fields<double>(location::vertex, isize, jsize, nhalo));
    m_y = (double *)malloc(
        size_of_mesh_fields<double>(location::vertex, isize, jsize, nhalo));
#endif

    for (size_t i = 0; i < isize; ++i) {
      for (size_t j = 0; j < jsize; ++j) {
        // cell to cell
        if (i > 0)
          m_cell_to_cells(i, 0, j, 0) = index(location::cell, i - 1, 1, j);
        m_cell_to_cells(i, 0, j, 1) = index(location::cell, i, 1, j);
        if (j > 0)
          m_cell_to_cells(i, 0, j, 2) = index(location::cell, i, 1, j - 1);

        if (i < isize - 1)
          m_cell_to_cells(i, 1, j, 0) = index(location::cell, i + 1, 0, j);
        m_cell_to_cells(i, 1, j, 1) = index(location::cell, i, 0, j);
        if (j < jsize - 1)
          m_cell_to_cells(i, 1, j, 2) = index(location::cell, i, 0, j + 1);

        // cell to edge
        m_cell_to_edges(i, 0, j, 0) = index(location::edge, i, 1, j);
        m_cell_to_edges(i, 0, j, 1) = index(location::edge, i, 2, j);
        m_cell_to_edges(i, 0, j, 2) = index(location::edge, i, 0, j);

        if (i < isize - 1)
          m_cell_to_edges(i, 1, j, 0) = index(location::edge, i + 1, 1, j);
        m_cell_to_edges(i, 1, j, 1) = index(location::edge, i, 2, j);
        if (j < jsize - 1)
          m_cell_to_edges(i, 1, j, 2) = index(location::edge, i, 0, j + 1);

        if (j < jsize - 1)
          m_cell_to_vertices(i, 0, j, 0) = index(location::vertex, i, 0, j + 1);

        if (i < isize - 1)
          m_cell_to_vertices(i, 0, j, 1) = index(location::vertex, i + 1, 0, j);
        m_cell_to_vertices(i, 0, j, 2) = index(location::vertex, i, 0, j);

        if (j < jsize - 1)
          m_cell_to_vertices(i, 1, j, 0) = index(location::vertex, i, 0, j + 1);

        if (j < jsize - 1 && i < isize - 1)
          m_cell_to_vertices(i, 1, j, 1) =
              index(location::vertex, i + 1, 0, j + 1);

        if (i < isize - 1)
          m_cell_to_vertices(i, 1, j, 2) = index(location::vertex, i + 1, 0, j);
        // edge to edge
        if (j > 0)
          m_edge_to_edges(i, 0, j, 0) = index(location::edge, i, 2, j - 1);
        m_edge_to_edges(i, 0, j, 1) = index(location::edge, i, 1, j);
        if (j > 0 && i < isize - 1)
          m_edge_to_edges(i, 0, j, 2) = index(location::edge, i + 1, 1, j - 1);
        m_edge_to_edges(i, 0, j, 3) = index(location::edge, i, 2, j);
        m_edge_to_edges(i, 1, j, 0) = index(location::edge, i, 0, j);
        if (i > 0)
          m_edge_to_edges(i, 1, j, 1) = index(location::edge, i - 1, 2, j);
        if (i > 0 && j < jsize - 1)
          m_edge_to_edges(i, 1, j, 2) = index(location::edge, i - 1, 0, j + 1);
        m_edge_to_edges(i, 1, j, 3) = index(location::edge, i, 2, j);
        m_edge_to_edges(i, 2, j, 0) = index(location::edge, i, 0, j);
        m_edge_to_edges(i, 2, j, 1) = index(location::edge, i, 1, j);
        if (i < isize - 1)
          m_edge_to_edges(i, 2, j, 2) = index(location::edge, i + 1, 1, j);
        if (j < jsize - 1)
          m_edge_to_edges(i, 2, j, 3) = index(location::edge, i, 0, j + 1);

        if (j > 0)
          m_edge_to_cells(i, 0, j, 0) = index(location::cell, i, 1, j - 1);
        m_edge_to_cells(i, 0, j, 1) = index(location::cell, i, 0, j);
        if (i > 0)
          m_edge_to_cells(i, 1, j, 0) = index(location::cell, i - 1, 1, j);
        m_edge_to_cells(i, 1, j, 1) = index(location::cell, i, 0, j);

        m_edge_to_cells(i, 2, j, 0) = index(location::cell, i, 1, j);
        m_edge_to_cells(i, 2, j, 1) = index(location::cell, i, 0, j);

        if (i < isize - 1)
          m_edge_to_vertices(i, 0, j, 0) = index(location::vertex, i + 1, 0, j);

        m_edge_to_vertices(i, 0, j, 1) = index(location::vertex, i, 0, j);

        m_edge_to_vertices(i, 1, j, 0) = index(location::vertex, i, 0, j);

        if (j < jsize - 1)
          m_edge_to_vertices(i, 1, j, 1) = index(location::vertex, i, 0, j + 1);

        if (j < jsize - 1)
          m_edge_to_vertices(i, 2, j, 0) = index(location::vertex, i, 0, j + 1);

        if (i < isize - 1)
          m_edge_to_vertices(i, 2, j, 1) = index(location::vertex, i + 1, 0, j);

        if (j > 0)
          m_vertex_to_vertices(i, 0, j, 0) =
              index(location::vertex, i, 0, j - 1);
        if (j < jsize - 1)
          m_vertex_to_vertices(i, 0, j, 1) =
              index(location::vertex, i, 0, j + 1);
        if (i < isize - 1)
          m_vertex_to_vertices(i, 0, j, 2) =
              index(location::vertex, i + 1, 0, j);
        if (i > 0)
          m_vertex_to_vertices(i, 0, j, 3) =
              index(location::vertex, i - 1, 0, j);
        if (i < isize - 1 && j > 0)
          m_vertex_to_vertices(i, 0, j, 4) =
              index(location::vertex, i + 1, 0, j - 1);
        if (i > 0 && j < jsize - 1)
          m_vertex_to_vertices(i, 0, j, 5) =
              index(location::vertex, i - 1, 0, j + 1);

        if (j > 0)
          m_vertex_to_edges(i, 0, j, 0) = index(location::edge, i, 1, j - 1);
        if (i > 0)
          m_vertex_to_edges(i, 0, j, 1) = index(location::edge, i - 1, 0, j);
        if (i > 0)
          m_vertex_to_edges(i, 0, j, 2) = index(location::edge, i - 1, 2, j);
        m_vertex_to_edges(i, 0, j, 3) = index(location::edge, i, 1, j);
        m_vertex_to_edges(i, 0, j, 4) = index(location::edge, i, 0, j);
        if (j > 0)
          m_vertex_to_edges(i, 0, j, 5) = index(location::edge, i, 2, j - 1);

        m_x[index(location::vertex, i, 0, j)] = i + j * 0.5;
        m_y[index(location::vertex, i, 0, j)] = j;

        //      }
        //    }

        //    for (size_t j = 0; j < m_jsize; ++j) {
        //      double x = m_isize + j * 0.5;
        //      double y = j;
        //      ss << end_idx_compute_domain() + j + 1 << " " << x << " " <<
        //      y
        //      << " 1"
      }
    }
    // add first line artificial nodes halo on the East
    m_curr_idx = end_idx_compute_domain();
    for (size_t j = 0; j < jsize; ++j) {
      m_cell_to_vertices(isize - 1, 0, j, 1) = m_curr_idx + j;

      m_cell_to_vertices(isize - 1, 1, j, 2) = m_curr_idx + j;

      if (j > 0)
        m_cell_to_vertices(isize - 1, 1, j - 1, 1) = m_curr_idx + j;

      m_edge_to_vertices(isize - 1, 0, j, 0) = m_curr_idx + j;
      m_edge_to_vertices(isize - 1, 2, j, 1) = m_curr_idx + j;
      m_x[m_curr_idx + j] = m_isize + j * 0.5;
      m_y[m_curr_idx + j] = j;
    }
    m_curr_idx += jsize;
    m_i_domain[1]++;
    m_j_domain[1]++;

    // add first line artificial nodes halo on the North
    for (int i = m_i_domain[0]; i < m_i_domain[1]; ++i) {
      std::cout << "KKB  " << i << " " << m_cell_to_vertices(0, 1, 5, 0) << " "
                << m_cell_to_vertices(0, 1, 5, 1) << " "
                << m_cell_to_vertices(0, 1, 5, 2) << std::endl;

      m_cell_to_vertices(i, 0, jsize - 1, 0) = m_curr_idx + i;

      m_cell_to_vertices(i, 1, jsize - 1, 0) = m_curr_idx + i;
      if (i > 0) {
        std::cout << "KKA  " << m_cell_to_vertices(0, 1, 5, 0) << " "
                  << m_cell_to_vertices(0, 1, 5, 1) << " "
                  << m_cell_to_vertices(0, 1, 5, 2) << std::endl;

        m_cell_to_vertices(i - 1, 1, jsize - 1, 1) = m_curr_idx + i;
        std::cout << "For " << i - 1 << " neigh " << m_curr_idx + i
                  << std::endl;

        std::cout << "KKG  " << m_cell_to_vertices(0, 1, 5, 0) << " "
                  << m_cell_to_vertices(0, 1, 5, 1) << " "
                  << m_cell_to_vertices(0, 1, 5, 2) << std::endl;
      }
      m_edge_to_vertices(i, 1, jsize - 1, 1) = m_curr_idx + i;
      m_edge_to_vertices(i, 2, jsize - 1, 0) = m_curr_idx + i;
      m_x[m_curr_idx + i] = i + m_jsize * 0.5;
      m_y[m_curr_idx + i] = m_jsize;
    }
    std::cout << "KKH  " << m_cell_to_vertices(0, 1, 5, 0) << " "
              << m_cell_to_vertices(0, 1, 5, 1) << " "
              << m_cell_to_vertices(0, 1, 5, 2) << std::endl;

    m_curr_idx += m_i_domain[1] - m_i_domain[0];
    //    m_j_domain[1]++;

    // add first line real halo on the West
    // add first line artificial nodes halo on the North
    for (int j = m_j_domain[0]; j < m_j_domain[1]; ++j) {
      m_cell_to_vertices(-1, 1, j, 0) = m_curr_idx + j + 1;
      m_cell_to_vertices(-1, 1, j, 1) = m_cell_to_vertices(0, 1, j, 0);
      m_cell_to_vertices(-1, 1, j, 2) = m_cell_to_vertices(0, 0, j, 2);

      m_cell_to_vertices(-1, 0, j, 0) = m_curr_idx + j + 1;
      m_cell_to_vertices(-1, 0, j, 1) = m_cell_to_vertices(0, 0, j, 2);
      m_cell_to_vertices(-1, 0, j, 2) = m_curr_idx + j;

      m_x[m_curr_idx + j] = j * 0.5 - 1;
      m_y[m_curr_idx + j] = j;
    }
    m_curr_idx += m_j_domain[1] - m_j_domain[0];
    m_j_domain[1]--;
  }

  size_t element_index(location primary_loc, int i, unsigned int c, int j) {
    assert(i > -(int)(m_nhalo) && i < (int)(m_isize + m_nhalo));
    assert(c < num_colors(primary_loc));
    assert(j > -(int)(m_nhalo) && j < (int)(m_jsize + m_nhalo));

    return (i + (int)m_nhalo) + c * (m_isize + m_nhalo * 2) +
           (j + (int)m_nhalo) * (m_isize + m_nhalo * 2) *
               num_colors(primary_loc);
  }

  size_t index(location primary_loc, unsigned int i, unsigned int c,
               unsigned int j) {
    assert(i < m_isize);
    assert(c < num_colors(primary_loc));
    assert(j < m_jsize);

    return i + c * (m_isize) + j * (m_isize)*num_colors(primary_loc);
  }

  void print() {
    std::stringstream ss;
    ss << "$MeshFormat" << std::endl
       << "2.2 0 8" << std::endl
       << "$EndMeshFormat" << std::endl;

    ss << "$Nodes" << std::endl;
    ss << m_curr_idx << std::endl; // num_nodes(m_isize + 1, m_jsize + 1, 0)
                                   // + 1 << std::endl;

    for (size_t i = 0; i < m_curr_idx; ++i) {
      ss << i + 1 << " " << m_x[i] << " " << m_y[i] << " 1 " << std::endl;
    }
    //    for (size_t j = 0; j < m_jsize; ++j) {
    //      for (size_t i = 0; i < m_isize; ++i) {
    //        ss << index(location::vertex, i, 0, j) + 1 << " "
    //           << m_x[index(location::vertex, i, 0, j)] << " "
    //           << m_y[index(location::vertex, i, 0, j)] << " 1" <<
    //           std::endl;
    //      }
    //    }

    //    for (size_t j = 0; j < m_jsize; ++j) {
    //      double x = m_isize + j * 0.5;
    //      double y = j;
    //      ss << end_idx_compute_domain() + j + 1 << " " << x << " " << y
    //      << "
    //      1 "
    //         << std::endl;
    //    }

    //    for (size_t i = 0; i < m_isize + 1; ++i) {
    //      double x = i + m_jsize * 0.5;
    //      double y = m_jsize;
    //      ss << GHOST_ID_Y + i + 1 << " " << x << " " << y << " 1" <<
    //      std::endl;
    //    }

    ss << "$EndNodes" << std::endl;
    ss << "$Elements" << std::endl;
    ss << num_nodes(m_isize + 1, m_jsize, 0) * 2 // edges
       // + num_nodes(m_isize, m_jsize, 0) * 3
       << std::endl;

    std::cout << "KK  " << m_cell_to_vertices(0, 1, 5, 0) << " "
              << m_cell_to_vertices(0, 1, 5, 1) << " "
              << m_cell_to_vertices(0, 1, 5, 2) << std::endl;
    for (size_t j = 0; j < m_jsize; ++j) {
      for (int i = -1; i < (int)m_isize; ++i) {
        ss << element_index(location::cell, i, 0, j) + 1 << " 2 4 1 1 1 28 "
           << m_cell_to_vertices(i, 0, j, 0) + 1 << " "
           << m_cell_to_vertices(i, 0, j, 1) + 1 << " "
           << m_cell_to_vertices(i, 0, j, 2) + 1 << std::endl;

        ss << element_index(location::cell, i, 1, j) + 1 << " 2 4 1 1 1 28 "
           << m_cell_to_vertices(i, 1, j, 0) + 1 << " "
           << m_cell_to_vertices(i, 1, j, 1) + 1 << " "
           << m_cell_to_vertices(i, 1, j, 2) + 1 << std::endl;
      }
    }

    //    // print the edges
    //    for (size_t j = 0; j < m_jsize; ++j) {
    //      for (size_t i = 0; i < m_isize; ++i) {
    //        ss << index(location::edge, i, 0, j) + 1 + 10000 << " 1 4 1 1
    //        1 28
    //        "
    //           << m_edge_to_vertices(i, 0, j, 0) + 1 << " "
    //           << m_edge_to_vertices(i, 0, j, 1) + 1 << std::endl;

    //        ss << index(location::edge, i, 1, j) + 1 + 10000 << " 1 4 1 1
    //        1 28
    //        "
    //           << m_edge_to_vertices(i, 1, j, 0) + 1 << " "
    //           << m_edge_to_vertices(i, 1, j, 1) + 1 << std::endl;
    //        ss << index(location::edge, i, 2, j) + 1 + 10000 << " 1 4 1  1
    //        1
    //        28 "
    //           << m_edge_to_vertices(i, 2, j, 0) + 1 << " "
    //           << m_edge_to_vertices(i, 2, j, 1) + 1 << std::endl;
    //      }
    //    }

    ss << "$EndElements" << std::endl;
    std::ofstream msh_file;
    msh_file.open("mesh.gmsh");
    msh_file << ss.str();
    msh_file.close();
  }

private:
  size_t m_isize, m_jsize, m_nhalo;
  std::array<int, 2> m_i_domain, m_j_domain;
  size_t m_curr_idx;
  double *m_x;
  double *m_y;

  neighbours_table m_cell_to_cells;
  neighbours_table m_cell_to_edges;
  neighbours_table m_cell_to_vertices;
  neighbours_table m_edge_to_cells;
  neighbours_table m_edge_to_edges;
  neighbours_table m_edge_to_vertices;
  neighbours_table m_vertex_to_cells;
  neighbours_table m_vertex_to_edges;
  neighbours_table m_vertex_to_vertices;
};
