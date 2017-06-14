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

#define GHOST_ID_X 10000
#define GHOST_ID_Y 20000

enum class location { cell = 0, edge, vertex };

constexpr unsigned int num_colors(location loc) {
  return loc == location::cell ? 2 : (loc == location::edge ? 3 : 1);
}

class mesh {

  static constexpr size_t num_nodes(const unsigned int isize,
                                    const unsigned int jsize) {
    return isize * jsize;
  }

  static constexpr size_t num_neighbours(const location primary_loc,
                                         const location neigh_loc) {
    return primary_loc == location::cell
               ? 3
               : (primary_loc == location::edge
                      ? (neigh_loc == location::edge ? 4 : 2)
                      : (6));
  }

  static constexpr size_t size_of_array(const location primary_loc,
                                        const location neigh_loc,
                                        const unsigned int isize,
                                        const unsigned jsize) {
    return num_nodes(isize, jsize) * num_colors(primary_loc) * sizeof(size_t) *
           num_neighbours(primary_loc, neigh_loc);
  }

public:
  mesh(const unsigned int isize, const unsigned int jsize)
      : m_isize(isize), m_jsize(jsize) {
#ifdef ENABLE_GPU
    cudaMallocManaged(
        &m_cell_to_cells,
        size_of_array(location::cell, location::cell, isize, jsize));
    cudaMallocManaged(
        &m_cell_to_edges,
        size_of_array(location::cell, location::edge, isize, jsize));
    cudaMallocManaged(
        &m_cell_to_vertices,
        size_of_array(location::cell, location::vertex, isize, jsize));
    cudaMallocManaged(
        &m_edge_to_cells,
        size_of_array(location::edge, location::cell, isize, jsize));
    cudaMallocManaged(
        &m_edge_to_edges,
        size_of_array(location::edge, location::edge, isize, jsize));
    cudaMallocManaged(
        &m_edge_to_vertices,
        size_of_array(location::edge, location::vertex, isize, jsize));
    cudaMallocManaged(
        &m_vertex_to_cells,
        size_of_array(location::vertex, location::cell, isize, jsize));
    cudaMallocManaged(
        &m_vertex_to_edges,
        size_of_array(location::vertex, location::edge, isize, jsize));
    cudaMallocManaged(
        &m_vertex_to_vertices,
        size_of_array(location::vertex, location::vertex, isize, jsize));
#else
    m_cell_to_cells = (size_t *)malloc(
        size_of_array(location::cell, location::cell, isize, jsize));
    m_cell_to_edges = (size_t *)malloc(
        size_of_array(location::cell, location::edge, isize, jsize));
    m_cell_to_vertices = (size_t *)malloc(
        size_of_array(location::cell, location::vertex, isize, jsize));
    m_edge_to_cells = (size_t *)malloc(
        size_of_array(location::edge, location::cell, isize, jsize));
    m_edge_to_edges = (size_t *)malloc(
        size_of_array(location::edge, location::edge, isize, jsize));
    m_edge_to_vertices = (size_t *)malloc(
        size_of_array(location::edge, location::vertex, isize, jsize));
    m_vertex_to_cells = (size_t *)malloc(
        size_of_array(location::vertex, location::cell, isize, jsize));
    m_vertex_to_edges = (size_t *)malloc(
        size_of_array(location::vertex, location::edge, isize, jsize));
    m_vertex_to_vertices = (size_t *)malloc(
        size_of_array(location::vertex, location::vertex, isize, jsize));
#endif

    for (size_t i = 0; i < isize; ++i) {
      for (size_t j = 0; j < jsize; ++j) {
        // cell to cell
        if (i > 0)
          m_cell_to_cells[index_in_tables(location::cell, location::cell, i, 0,
                                          j, 0)] =
              index(location::cell, i - 1, 1, j);
        m_cell_to_cells[index_in_tables(location::cell, location::cell, i, 0, j,
                                        1)] = index(location::cell, i, 1, j);
        if (j > 0)
          m_cell_to_cells[index_in_tables(location::cell, location::cell, i, 0,
                                          j, 2)] =
              index(location::cell, i, 1, j - 1);

        if (i < isize - 1)
          m_cell_to_cells[index_in_tables(location::cell, location::cell, i, 1,
                                          j, 0)] =
              index(location::cell, i + 1, 0, j);
        m_cell_to_cells[index_in_tables(location::cell, location::cell, i, 1, j,
                                        1)] = index(location::cell, i, 0, j);
        if (j < jsize - 1)
          m_cell_to_cells[index_in_tables(location::cell, location::cell, i, 1,
                                          j, 2)] =
              index(location::cell, i, 0, j + 1);

        // cell to edge
        m_cell_to_edges[index_in_tables(location::cell, location::edge, i, 0, j,
                                        0)] = index(location::edge, i, 1, j);
        m_cell_to_edges[index_in_tables(location::cell, location::edge, i, 0, j,
                                        1)] = index(location::edge, i, 2, j);
        m_cell_to_edges[index_in_tables(location::cell, location::edge, i, 0, j,
                                        2)] = index(location::edge, i, 0, j);

        if (i < isize - 1)
          m_cell_to_edges[index_in_tables(location::cell, location::edge, i, 1,
                                          j, 0)] =
              index(location::edge, i + 1, 1, j);
        m_cell_to_edges[index_in_tables(location::cell, location::edge, i, 1, j,
                                        1)] = index(location::edge, i, 2, j);
        if (j < jsize - 1)
          m_cell_to_edges[index_in_tables(location::cell, location::edge, i, 1,
                                          j, 2)] =
              index(location::edge, i, 0, j + 1);

        m_cell_to_vertices[index_in_tables(location::cell, location::vertex, i,
                                           0, j, 0)] =
            (j < jsize - 1) ? index(location::vertex, i, 0, j + 1)
                            : GHOST_ID_Y + i;
        m_cell_to_vertices[index_in_tables(location::cell, location::vertex, i,
                                           0, j, 1)] =
            (i < isize - 1) ? index(location::vertex, i + 1, 0, j)
                            : GHOST_ID_X + j;
        m_cell_to_vertices[index_in_tables(location::cell, location::vertex, i,
                                           0, j, 2)] =
            index(location::vertex, i, 0, j);

        m_cell_to_vertices[index_in_tables(location::cell, location::vertex, i,
                                           1, j, 0)] =
            (j < jsize - 1) ? index(location::vertex, i, 0, j + 1)
                            : GHOST_ID_Y + i;

        m_cell_to_vertices[index_in_tables(location::cell, location::vertex, i,
                                           1, j, 1)] =
            (j < jsize - 1 && i < isize - 1)
                ? index(location::vertex, i + 1, 0, j + 1)
                : (j < jsize - 1 ? GHOST_ID_X + j + 1 : GHOST_ID_Y + i + 1);

        m_cell_to_vertices[index_in_tables(location::cell, location::vertex, i,
                                           1, j, 2)] =
            (i < isize - 1) ? index(location::vertex, i + 1, 0, j)
                            : GHOST_ID_X + j;
        // edge to edge
        if (j > 0)
          m_edge_to_edges[index_in_tables(location::edge, location::edge, i, 0,
                                          j, 0)] =
              index(location::edge, i, 2, j - 1);
        m_edge_to_edges[index_in_tables(location::edge, location::edge, i, 0, j,
                                        1)] = index(location::edge, i, 1, j);
        if (j > 0 && i < isize - 1)
          m_edge_to_edges[index_in_tables(location::edge, location::edge, i, 0,
                                          j, 2)] =
              index(location::edge, i + 1, 1, j - 1);
        m_edge_to_edges[index_in_tables(location::edge, location::edge, i, 0, j,
                                        3)] = index(location::edge, i, 2, j);
        m_edge_to_edges[index_in_tables(location::edge, location::edge, i, 1, j,
                                        0)] = index(location::edge, i, 0, j);
        if (i > 0)
          m_edge_to_edges[index_in_tables(location::edge, location::edge, i, 1,
                                          j, 1)] =
              index(location::edge, i - 1, 2, j);
        if (i > 0 && j < jsize - 1)
          m_edge_to_edges[index_in_tables(location::edge, location::edge, i, 1,
                                          j, 2)] =
              index(location::edge, i - 1, 0, j + 1);
        m_edge_to_edges[index_in_tables(location::edge, location::edge, i, 1, j,
                                        3)] = index(location::edge, i, 2, j);
        m_edge_to_edges[index_in_tables(location::edge, location::edge, i, 2, j,
                                        0)] = index(location::edge, i, 0, j);
        m_edge_to_edges[index_in_tables(location::edge, location::edge, i, 2, j,
                                        1)] = index(location::edge, i, 1, j);
        if (i < isize - 1)
          m_edge_to_edges[index_in_tables(location::edge, location::edge, i, 2,
                                          j, 2)] =
              index(location::edge, i + 1, 1, j);
        if (j < jsize - 1)
          m_edge_to_edges[index_in_tables(location::edge, location::edge, i, 2,
                                          j, 3)] =
              index(location::edge, i, 0, j + 1);

        if (j > 0)
          m_edge_to_cells[index_in_tables(location::edge, location::cell, i, 0,
                                          j, 0)] =
              index(location::cell, i, 1, j - 1);
        m_edge_to_cells[index_in_tables(location::edge, location::cell, i, 0, j,
                                        1)] = index(location::cell, i, 0, j);
        if (i > 0)
          m_edge_to_cells[index_in_tables(location::edge, location::cell, i, 1,
                                          j, 0)] =
              index(location::cell, i - 1, 1, j);
        m_edge_to_cells[index_in_tables(location::edge, location::cell, i, 1, j,
                                        1)] = index(location::cell, i, 0, j);

        m_edge_to_cells[index_in_tables(location::edge, location::cell, i, 2, j,
                                        0)] = index(location::cell, i, 1, j);
        m_edge_to_cells[index_in_tables(location::edge, location::cell, i, 2, j,
                                        1)] = index(location::cell, i, 0, j);

        m_edge_to_vertices[index_in_tables(location::edge, location::vertex, i,
                                           0, j, 0)] =
            (i < isize - 1) ? index(location::vertex, i + 1, 0, j)
                            : GHOST_ID_X + j;

        m_edge_to_vertices[index_in_tables(location::edge, location::vertex, i,
                                           0, j, 1)] =
            index(location::vertex, i, 0, j);

        m_edge_to_vertices[index_in_tables(location::edge, location::vertex, i,
                                           1, j, 0)] =
            index(location::vertex, i, 0, j);

        m_edge_to_vertices[index_in_tables(location::edge, location::vertex, i,
                                           1, j, 1)] =
            (j < jsize - 1) ? index(location::vertex, i, 0, j + 1)
                            : GHOST_ID_Y + i;

        m_edge_to_vertices[index_in_tables(location::edge, location::vertex, i,
                                           2, j, 0)] =
            (j < jsize - 1) ? index(location::vertex, i, 0, j + 1)
                            : GHOST_ID_Y + i;
        m_edge_to_vertices[index_in_tables(location::edge, location::vertex, i,
                                           2, j, 1)] =
            (i < isize - 1) ? index(location::vertex, i + 1, 0, j)
                            : GHOST_ID_X + j;

        if (j > 0)
          m_vertex_to_vertices[index_in_tables(location::vertex,
                                               location::vertex, i, 0, j, 0)] =
              index(location::vertex, i, 0, j - 1);
        if (j < jsize - 1)
          m_vertex_to_vertices[index_in_tables(location::vertex,
                                               location::vertex, i, 0, j, 1)] =
              index(location::vertex, i, 0, j + 1);
        if (i < isize - 1)
          m_vertex_to_vertices[index_in_tables(location::vertex,
                                               location::vertex, i, 0, j, 2)] =
              index(location::vertex, i + 1, 0, j);
        if (i > 0)
          m_vertex_to_vertices[index_in_tables(location::vertex,
                                               location::vertex, i, 0, j, 3)] =
              index(location::vertex, i - 1, 0, j);
        if (i < isize - 1 && j > 0)
          m_vertex_to_vertices[index_in_tables(location::vertex,
                                               location::vertex, i, 0, j, 4)] =
              index(location::vertex, i + 1, 0, j - 1);
        if (i > 0 && j < jsize - 1)
          m_vertex_to_vertices[index_in_tables(location::vertex,
                                               location::vertex, i, 0, j, 5)] =
              index(location::vertex, i - 1, 0, j + 1);

        if (j > 0)
          m_vertex_to_edges[index_in_tables(location::vertex, location::edge, i,
                                            0, j, 0)] =
              index(location::edge, i, 1, j - 1);
        if (i > 0)
          m_vertex_to_edges[index_in_tables(location::vertex, location::edge, i,
                                            0, j, 1)] =
              index(location::edge, i - 1, 0, j);
        if (i > 0)
          m_vertex_to_edges[index_in_tables(location::vertex, location::edge, i,
                                            0, j, 2)] =
              index(location::edge, i - 1, 2, j);
        m_vertex_to_edges[index_in_tables(location::vertex, location::edge, i,
                                          0, j, 3)] =
            index(location::edge, i, 1, j);
        m_vertex_to_edges[index_in_tables(location::vertex, location::edge, i,
                                          0, j, 4)] =
            index(location::edge, i, 0, j);
        if (j > 0)
          m_vertex_to_edges[index_in_tables(location::vertex, location::edge, i,
                                            0, j, 5)] =
              index(location::edge, i, 2, j - 1);
      }
    }
  }
  size_t index_in_tables(location primary_loc, location neigh_loc,
                         unsigned int i, unsigned int c, unsigned int j,
                         unsigned int neigh_idx) {
    assert(i < m_isize);
    assert(c < num_colors(primary_loc));
    assert(j < m_jsize);
    assert(neigh_idx < num_neighbours(primary_loc, neigh_loc));
    return i + c * m_isize + j * m_isize * num_colors(primary_loc) +
           neigh_idx * m_isize * num_colors(primary_loc) * m_jsize;
  }

  size_t index(location primary_loc, unsigned int i, unsigned int c,
               unsigned int j) {
    assert(i < m_isize);
    assert(c < num_colors(primary_loc));
    assert(j < m_jsize);

    return i + c * m_isize + j * m_isize * num_colors(primary_loc);
  }

  void print() {
    std::stringstream ss;
    ss << "$MeshFormat" << std::endl
       << "2.2 0 8" << std::endl
       << "$EndMeshFormat" << std::endl;

    ss << "$Nodes" << std::endl;
    ss << num_nodes(m_isize, m_jsize) + m_jsize + m_isize + 1 << std::endl;

    for (size_t j = 0; j < m_jsize; ++j) {
      for (size_t i = 0; i < m_isize; ++i) {
        double x = i + j * 0.5;
        double y = j;
        ss << index(location::vertex, i, 0, j) + 1 << " " << x << " " << y
           << " 1" << std::endl;
      }
    }

    for (size_t j = 0; j < m_jsize; ++j) {
      double x = m_isize + j * 0.5;
      double y = j;
      ss << GHOST_ID_X + j + 1 << " " << x << " " << y << " 1" << std::endl;
    }
    for (size_t i = 0; i < m_isize + 1; ++i) {
      double x = i + m_jsize * 0.5;
      double y = m_jsize;
      ss << GHOST_ID_Y + i + 1 << " " << x << " " << y << " 1" << std::endl;
    }

    ss << "$EndNodes" << std::endl;
    ss << "$Elements" << std::endl;
    ss << num_nodes(m_isize, m_jsize) * 2 + num_nodes(m_isize, m_jsize) * 3
       << std::endl;

    for (size_t j = 0; j < m_jsize; ++j) {
      for (size_t i = 0; i < m_isize; ++i) {
        ss << index(location::cell, i, 0, j) + 1 << " 2 4 1 1 1 28 "
           << m_cell_to_vertices[index_in_tables(
                  location::cell, location::vertex, i, 0, j, 0)] +
                  1
           << " "
           << m_cell_to_vertices[index_in_tables(
                  location::cell, location::vertex, i, 0, j, 1)] +
                  1
           << " "
           << m_cell_to_vertices[index_in_tables(
                  location::cell, location::vertex, i, 0, j, 2)] +
                  1
           << std::endl;

        ss << index(location::cell, i, 1, j) + 1 << " 2 4 1 1 1 28 "
           << m_cell_to_vertices[index_in_tables(
                  location::cell, location::vertex, i, 1, j, 0)] +
                  1
           << " "
           << m_cell_to_vertices[index_in_tables(
                  location::cell, location::vertex, i, 1, j, 1)] +
                  1
           << " "
           << m_cell_to_vertices[index_in_tables(
                  location::cell, location::vertex, i, 1, j, 2)] +
                  1
           << std::endl;
      }
    }

    // print the edges
    for (size_t j = 0; j < m_jsize; ++j) {
      for (size_t i = 0; i < m_isize; ++i) {
        ss << index(location::edge, i, 0, j) + 1 + 10000 << " 1 4 1 1 1 28 "
           << m_edge_to_vertices[index_in_tables(
                  location::edge, location::vertex, i, 0, j, 0)] +
                  1
           << " "
           << m_edge_to_vertices[index_in_tables(
                  location::edge, location::vertex, i, 0, j, 1)] +
                  1
           << std::endl;

        ss << index(location::edge, i, 1, j) + 1 + 10000 << " 1 4 1 1 1 28 "
           << m_edge_to_vertices[index_in_tables(
                  location::edge, location::vertex, i, 1, j, 0)] +
                  1
           << " "
           << m_edge_to_vertices[index_in_tables(
                  location::edge, location::vertex, i, 1, j, 1)] +
                  1
           << std::endl;
        ss << index(location::edge, i, 2, j) + 1 + 10000 << " 1 4 1  1 1 28 "
           << m_edge_to_vertices[index_in_tables(
                  location::edge, location::vertex, i, 2, j, 0)] +
                  1
           << " "
           << m_edge_to_vertices[index_in_tables(
                  location::edge, location::vertex, i, 2, j, 1)] +
                  1
           << std::endl;
      }
    }

    ss << "$EndElements" << std::endl;
    std::ofstream msh_file;
    msh_file.open("mesh.gmsh");
    msh_file << ss.str();
    msh_file.close();
  }

private:
  size_t m_isize, m_jsize;
  size_t *m_cell_to_cells;
  size_t *m_cell_to_edges;
  size_t *m_cell_to_vertices;
  size_t *m_edge_to_cells;
  size_t *m_edge_to_edges;
  size_t *m_edge_to_vertices;
  size_t *m_vertex_to_cells;
  size_t *m_vertex_to_edges;
  size_t *m_vertex_to_vertices;
};
