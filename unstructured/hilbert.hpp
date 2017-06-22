#pragma once

void hilbert(std::vector<std::array<int, 2>> &inds, double x, double y,
             double xi, double xj, double yi, double yj, double n) {

  if (n <= 0) {
    inds.push_back({(int)std::round(x + (xi + yi) / 2 - 0.5),
                    (int)std::round(y + (xj + yj) / 2 - 0.5)});
  } else {
    hilbert(inds, x, y, yi / 2, yj / 2, xi / 2, xj / 2, n - 1);
    hilbert(inds, x + xi / 2, y + xj / 2, xi / 2, xj / 2, yi / 2, yj / 2,
            n - 1);
    hilbert(inds, x + xi / 2 + yi / 2, y + xj / 2 + yj / 2, xi / 2, xj / 2,
            yi / 2, yj / 2, n - 1);
    hilbert(inds, x + xi / 2 + yi, y + xj / 2 + yj, -yi / 2, -yj / 2, -xi / 2,
            -xj / 2, n - 1);
  }
}
