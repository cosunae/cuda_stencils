#pragma once
#include <math.h>
#include <unordered_map>
#include "mesh.hpp"
#include "umesh.hpp"
#include "hilbert.hpp"

void mesh_to_hilbert(umesh &umesh_, mesh &mesh_) {

  assert(mesh_.isize() == mesh_.jsize());

  const double number_div = log2((float)mesh_.isize());
  double intpart;
  assert(std::modf(number_div, &intpart) == 0.0);

  std::vector<std::array<int, 2>> inds;
  std::cout << "*CALLING " << mesh_.isize() << " " << mesh_.jsize() << " "
            << number_div << std::endl;
  hilbert(inds, 0, 0, mesh_.isize(), 0, 0, mesh_.jsize(), number_div);

  size_t halo_idx = mesh_.compd_size();
  std::unordered_map<size_t, size_t> halo_idxs_pairs;

  size_t isize = mesh_.isize();
  size_t jsize = mesh_.jsize();

  for (int i = 0; i < inds.size(); ++i) {
    std::cout << " p " << inds[i][0] << " " << inds[i][1] << std::endl;
  }
  for (size_t idx = 0; idx != inds.size(); ++idx) {
    int i = inds[idx][0];
    int j = inds[idx][1];

    auto cells = umesh_.get_elements(location::cell);
    // color 0
    if (i > 0) {
      auto pos =
          std::find(inds.begin(), inds.end(), std::array<int, 2>{i - 1, j});
      assert(pos != std::end(inds));

      cells.table(location::cell)(idx * 2, 0) =
          std::distance(inds.begin(), pos) + 1;

    } else {
      cells.table(location::cell)(idx * 2, 0) = halo_idx;
      halo_idxs_pairs[halo_idx] = mesh_.get_elements(location::cell)
                                      .neighbor(location::cell, i - 1, 0, j, 0);
      halo_idx++;
    }

    {
      auto pos = std::find(inds.begin(), inds.end(), std::array<int, 2>{i, j});
      assert(pos != std::end(inds));

      cells.table(location::cell)(idx * 2, 1) =
          std::distance(inds.begin(), pos) + 1;
    }
    if (j > 0) {
      auto pos =
          std::find(inds.begin(), inds.end(), std::array<int, 2>{i, j - 1});
      assert(pos != std::end(inds));

      cells.table(location::cell)(idx * 2, 2) =
          std::distance(inds.begin(), pos) + 1;
    } else {
      cells.table(location::cell)(idx * 2, 2) = halo_idx;
      halo_idxs_pairs[halo_idx] = mesh_.get_elements(location::cell)
                                      .neighbor(location::cell, i, 0, j - 1, 2);
      halo_idx++;
    }

    // color 1
    if (i < isize - 1) {
      std::cout << "I " << i << std::endl;
      auto pos =
          std::find(inds.begin(), inds.end(), std::array<int, 2>{i + 1, j});
      assert(pos != std::end(inds));

      cells.table(location::cell)(idx * 2 + 1, 0) =
          std::distance(inds.begin(), pos);
    } else {
      cells.table(location::cell)(idx * 2 + 1, 0) = halo_idx;
      halo_idxs_pairs[halo_idx] = mesh_.get_elements(location::cell)
                                      .neighbor(location::cell, i + 1, 1, j, 0);
      halo_idx++;
    }

    {
      auto pos = std::find(inds.begin(), inds.end(), std::array<int, 2>{i, j});
      assert(pos != std::end(inds));

      cells.table(location::cell)(idx * 2 + 1, 1) =
          std::distance(inds.begin(), pos);
    }
    if (j < jsize - 1) {
      auto pos =
          std::find(inds.begin(), inds.end(), std::array<int, 2>{i, j + 1});
      assert(pos != std::end(inds));

      cells.table(location::cell)(idx * 2 + 1, 2) =
          std::distance(inds.begin(), pos);
    } else {
      cells.table(location::cell)(idx * 2 + 1, 2) = halo_idx;
      halo_idxs_pairs[halo_idx] = mesh_.get_elements(location::cell)
                                      .neighbor(location::cell, i, 1, j + 1, 2);
      halo_idx++;
    }

    for (size_t n = 0; n < num_neighbours(location::cell, location::vertex);
         ++n) {
      cells.table(location::vertex)(idx * 2, n) =
          mesh_.get_elements(location::cell)
              .neighbor(location::vertex, i, 0, j, n);
      cells.table(location::vertex)(idx * 2 + 1, n) =
          mesh_.get_elements(location::cell)
              .neighbor(location::vertex, i, 1, j, n);
    }
  }

  for (size_t cnt = 0; cnt != mesh_.get_nodes().totald_size(); ++cnt) {
    umesh_.get_nodes().x(cnt) = mesh_.get_nodes().x(cnt);
    umesh_.get_nodes().y(cnt) = mesh_.get_nodes().y(cnt);
  }

  //  for (size_t i = 0; i < m_isize; ++i) {
  //    for (size_t j = 0; j < m_jsize; ++j) {
  //    }
  //  }
}
