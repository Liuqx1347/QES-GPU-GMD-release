/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/**
 * @file LocalMixingSerial.cpp
 * @brief :document this:
 * @sa LocalMixing
 */

#include "LocalMixingSerial.h"
#ifdef HAS_CUDA_SUPPORT
#include "LocalMixingSerialGPU.h"
#endif
// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"
#include "LocalMixingNetCDF.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>
#include <chrono>

namespace {
inline void updateMinDistance(double &current, double dx, double dy, double dz)
{
  // Exact pruning: if any single coordinate difference is already >= current,
  // the full Euclidean distance cannot be smaller than current.
  const double adx = std::abs(dx);
  if (adx >= current) return;
  const double ady = std::abs(dy);
  if (ady >= current) return;
  const double adz = std::abs(dz);
  if (adz >= current) return;

  const double cand2 = dx * dx + dy * dy + dz * dz;
  const double curr2 = current * current;
  if (cand2 < curr2) {
    current = std::sqrt(cand2);
  }
}

inline void updateMinDistance2D(double &current, double dx, double dy)
{
  const double adx = std::abs(dx);
  if (adx >= current) return;
  const double ady = std::abs(dy);
  if (ady >= current) return;

  const double cand2 = dx * dx + dy * dy;
  const double curr2 = current * current;
  if (cand2 < curr2) {
    current = std::sqrt(cand2);
  }
}
} // namespace

void LocalMixingSerial::defineMixingLength(const WINDSInputData *WID, WINDSGeneralData *WGD)
{
  // Fast path for repeated runs on the same geometry:
  // if a cached NetCDF file already exists, load it directly and skip the expensive serial build.
  if (WID->turbParams->save2file && !WID->turbParams->filename.empty()) {
    std::ifstream fin(WID->turbParams->filename.c_str());
    if (fin.good()) {
      fin.close();
      std::cout << "[MixLength] \t cache found, loading mixing length from "
                << WID->turbParams->filename << std::endl;
      LocalMixingNetCDF cacheLoader;
      cacheLoader.defineMixingLength(WID, WGD);
      return;
    }
  }

  const int nx = WGD->nx;
  const int ny = WGD->ny;
  const int nz = WGD->nz;

  const float dz = WGD->dz;
  const float dy = WGD->dy;
  const float dx = WGD->dx;

  const int nxc = nx - 1;
  const int nyc = ny - 1;
  const int plane = nxc * nyc;

  // x-grid (face-center & cell-center)
  x_fc.resize(nx, 0.0f);
  x_cc.resize(nx - 1, 0.0f);

  // y-grid (face-center & cell-center)
  y_fc.resize(ny, 0.0f);
  y_cc.resize(ny - 1, 0.0f);

  // z-grid (face-center & cell-center)
  z_fc.resize(nz, 0.0f);
  z_cc.resize(nz - 1, 0.0f);

  // x cell-center
  x_cc = WGD->x;
  // x face-center (this assume constant dx for the moment, same as QES-winds)
  for (int i = 1; i < nx - 1; ++i) {
    x_fc[i] = 0.5f * (WGD->x[i - 1] + WGD->x[i]);
  }
  x_fc[0] = x_fc[1] - dx;
  x_fc[nx - 1] = x_fc[nx - 2] + dx;

  // y cell-center
  y_cc = WGD->y;
  // y face-center (this assume constant dy for the moment, same as QES-winds)
  for (int i = 1; i < ny - 1; ++i) {
    y_fc[i] = 0.5f * (WGD->y[i - 1] + WGD->y[i]);
  }
  y_fc[0] = y_fc[1] - dy;
  y_fc[ny - 1] = y_fc[ny - 2] + dy;

  // z cell-center
  z_cc = WGD->z;
  // z face-center (with ghost cell under the ground)
  for (int i = 1; i < nz; ++i) {
    z_fc[i] = WGD->z_face[i];
  }
  z_fc[0] = z_fc[1] - dz;

  // find max height of solid objects in the domain
  float max_z = 0.0f;
  for (int i = 1; i < nx - 2; ++i) {
    for (int j = 1; j < ny - 2; ++j) {
      for (int k = 0; k < nz - 2; ++k) {
        const int icell_cent = i + j * nxc + k * plane;
        if ((WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2) && max_z < z_cc[k]) {
          max_z = z_cc[k];
        }
      }
    }
  }

  // maximum height of local mixing length = 3*max z of objects
  int max_height = nz - 2;
  for (int k = 0; k < nz - 1; ++k) {
    if (z_cc[k] > 3.0f * max_z) {
      max_height = k;
      break;
    }
  }

  // seed local mixing length with the vertical distance to the terrain
  for (int i = 1; i < nx - 2; ++i) {
    for (int j = 1; j < ny - 2; ++j) {
      const int terrainId = i + j * nxc;
      for (int k = 0; k < nz - 2; ++k) {
        const int icell_cent = i + j * nxc + k * plane;
        if ((WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2)) {
          if (k < max_height) {
            WGD->mixingLengths[icell_cent] = z_cc[k] - WGD->terrain[terrainId];
          } else {
            WGD->mixingLengths[icell_cent] = z_cc[k];
          }
          if (WGD->mixingLengths[icell_cent] < 0.0f) {
            WGD->mixingLengths[icell_cent] = 0.0f;
          }
        }
      }
    }
  }

  getMinDistWall(WGD, max_height);

  // linear interpolation between 3.0*max_z and 3.6*max_z
  std::cout << "[MixLength] \t linear interp of mixing length" << std::endl;
  const int k1 = std::min(max_height - 1, nz - 2);
  const int k2 = std::min(k1 + k1 / 5, nz - 2);
  for (int i = 1; i < nx - 2; ++i) {
    for (int j = 1; j < ny - 2; ++j) {
      const int id1 = i + j * nxc + k1 * plane;
      const int id2 = i + j * nxc + k2 * plane;
      const float slope = (WGD->mixingLengths[id2] - WGD->mixingLengths[id1]) / (z_cc[k2] - z_cc[k1]);
      for (int k = k1; k < k2; ++k) {
        const int id_cc = i + j * nxc + k * plane;
        WGD->mixingLengths[id_cc] = WGD->mixingLengths[id1] + (z_cc[k] - z_cc[k1]) * slope;
      }
    }
  }

  if (WID->turbParams->save2file) {
    saveMixingLength(WID, WGD);
  }
}

void LocalMixingSerial::getMinDistWall(WINDSGeneralData *WGD, int max_height)
{
  const int nx = WGD->nx;
  const int ny = WGD->ny;
  const int nz = WGD->nz;

  const int nxc = nx - 1;
  const int nyc = ny - 1;
  const int plane = nxc * nyc;

  // Use local vectors here: this function is called once per initialization, and local vectors
  // avoid stale state while letting us reserve capacity explicitly.
  std::vector<int> wall_below_work;
  std::vector<int> wall_back_work;
  std::vector<int> wall_front_work;
  std::vector<int> wall_right_work;
  std::vector<int> wall_left_work;

  const size_t approxFluid = static_cast<size_t>(std::max(1, nx - 3))
                           * static_cast<size_t>(std::max(1, ny - 3))
                           * static_cast<size_t>(std::max(1, max_height));
  wall_below_work.reserve(approxFluid / 4 + 1024);
  wall_back_work.reserve(approxFluid / 32 + 1024);
  wall_front_work.reserve(approxFluid / 32 + 1024);
  wall_right_work.reserve(approxFluid / 32 + 1024);
  wall_left_work.reserve(approxFluid / 32 + 1024);

  size_t wall_below_total_count = 0;
  size_t wall_below_terrain_only_count = 0;
  size_t wall_above_count = 0;

  // define the walls. For the "wall below" case, terrain-only cells are handled immediately
  // and skipped from the expensive propagation loop because the old code ended up doing nothing
  // for them after checking the 4 terrain corners.
  for (int i = 1; i < nx - 2; ++i) {
    for (int j = 1; j < ny - 2; ++j) {
      for (int k = 1; k < nz - 2; ++k) {
        const int icell_cent = i + j * nxc + k * plane;

        if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2) {
          continue;
        }

        // Wall below
    if (WGD->icellflag[icell_cent - plane] == 0 || WGD->icellflag[icell_cent - plane] == 2) {

       // 这个是和 CPU 原始版可比的总 wall-below active cell 数量
      ++wall_below_total_count;

  const int idxp = icell_cent - plane + 1;
  const int idxm = icell_cent - plane - 1;
  const int idyp = icell_cent - plane + nxc;
  const int idym = icell_cent - plane - nxc;

  const bool terrainAll =
    (WGD->icellflag[idxp] == 2 && WGD->icellflag[idxm] == 2 &&
     WGD->icellflag[idyp] == 2 && WGD->icellflag[idym] == 2);

  if (terrainAll) {
    // 这个 cell 是纯地形上方 cell，CPU 原始版对它只做本地赋值，不做传播
    ++wall_below_terrain_only_count;

    // 为了严格对齐 CPU 原始版，这里用 =，不要用 std::min
    WGD->mixingLengths[icell_cent] = static_cast<double>(z_cc[k] - z_fc[k]);

  } else {
    // 这些才是真正需要 GPU 计算传播的 wall-below cell
    wall_below_work.push_back(icell_cent);
  }
}

        // Wall above (only counted for diagnostics in the current algorithm)
        if (WGD->icellflag[icell_cent + plane] == 0 || WGD->icellflag[icell_cent + plane] == 2) {
          ++wall_above_count;
        }

        // Wall in back
        if (WGD->icellflag[icell_cent - 1] == 0 || WGD->icellflag[icell_cent - 1] == 2) {
          wall_back_work.push_back(icell_cent);
        }

        // Wall in front
        if (WGD->icellflag[icell_cent + 1] == 0 || WGD->icellflag[icell_cent + 1] == 2) {
          wall_front_work.push_back(icell_cent);
        }

        // Wall on right
        if (WGD->icellflag[icell_cent - nxc] == 0 || WGD->icellflag[icell_cent - nxc] == 2) {
          wall_right_work.push_back(icell_cent);
        }

        // Wall on left
        if (WGD->icellflag[icell_cent + nxc] == 0 || WGD->icellflag[icell_cent + nxc] == 2) {
          wall_left_work.push_back(icell_cent);
        }
      }
    }
  }

  std::cout << "[MixLength] \t active cells with wall below: " << wall_below_total_count << std::endl;
  std::cout << "[MixLength] \t terrain-only active cells with wall below: " << wall_below_terrain_only_count << std::endl;
  std::cout << "[MixLength] \t GPU work cells with wall below: " << wall_below_work.size() << std::endl;
  std::cout << "[MixLength] \t cells with wall above: " << wall_above_count << std::endl;
  std::cout << "[MixLength] \t cells with wall in the front: " << wall_front_work.size() << std::endl;
  std::cout << "[MixLength] \t cells with wall in the back: " << wall_back_work.size() << std::endl;
  std::cout << "[MixLength] \t cells with wall to the right: " << wall_right_work.size() << std::endl;
  std::cout << "[MixLength] \t cells with wall to the left: " << wall_left_work.size() << std::endl;

  auto tBelowStart = std::chrono::high_resolution_clock::now();

  bool wallBelowDoneOnGPU = false;

#ifdef HAS_CUDA_SUPPORT
  if (!wall_below_work.empty()) {
    const auto wallBelowGpuStart = std::chrono::high_resolution_clock::now();

    wallBelowDoneOnGPU = localMixingWallBelowGPU(
        WGD,
        max_height,
        x_cc,
        y_cc,
        z_cc,
        z_fc,
        wall_below_work);

    const auto wallBelowGpuEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> wallBelowGpuElapsed = wallBelowGpuEnd - wallBelowGpuStart;

    if (wallBelowDoneOnGPU) {
     std::cout << "[MixLength] \t GPU work cells with wall below: DONE on GPU, elapsed time: "
          << wallBelowGpuElapsed.count() << " s" << std::endl;
    } else {
      std::cout << "[MixLength] \t wall below GPU failed, fallback to CPU" << std::endl;
    }
  }
#endif

  if (!wallBelowDoneOnGPU) {

    // Apply mixing length to the cells with wall below
    for (size_t id = 0; id < wall_below_work.size(); ++id) {
      const int id_cc = wall_below_work[id];
      const int idxp = id_cc - plane + 1;
      const int idxm = id_cc - plane - 1;
      const int idyp = id_cc - plane + nxc;
      const int idym = id_cc - plane - nxc;

      const int k = static_cast<int>(id_cc / plane);
      const int j = static_cast<int>((id_cc - k * plane) / nxc);
      const int i = id_cc - j * nxc - k * plane;
      const int maxdist = max_height - k;

      WGD->mixingLengths[id_cc] = static_cast<double>(z_cc[k] - z_fc[k]);

      const float x1 = x_cc[i];
      const float y1 = y_cc[j];
      const float z1 = z_fc[k];

      if (WGD->icellflag[idxp] == 0 && WGD->icellflag[idxm] == 0 &&
          WGD->icellflag[idyp] == 0 && WGD->icellflag[idym] == 0) {
        // Building on all 4 corners -> propagate vertically only.
        for (int kk = 0; kk <= maxdist; ++kk) {
          const int kkAbs = kk + k;
          const int idDst = i + j * nxc + kkAbs * plane;
          const float dzv = z_cc[kkAbs] - z1;
          if (dzv < WGD->mixingLengths[idDst]) {
            WGD->mixingLengths[idDst] = dzv;
          }
        }
      } else {
        // General case: propagate in all directions.
        for (int kk = 0; kk <= maxdist; ++kk) {
          const int kkAbs = kk + k;
          const float dzv = z_cc[kkAbs] - z1;

          const int reach = maxdist + kk;
          const int i1 = std::max(i - reach, 1);
          const int i2 = std::min(i + reach, nx - 2);
          const int j1 = std::max(j - reach, 1);
          const int j2 = std::min(j + reach, ny - 2);

          for (int jj = j1; jj <= j2; ++jj) {
            const float dyv = y_cc[jj] - y1;
            for (int ii = i1; ii <= i2; ++ii) {
              const int idDst = ii + jj * nxc + kkAbs * plane;
              const float dxv = x_cc[ii] - x1;
              updateMinDistance(WGD->mixingLengths[idDst], dxv, dyv, dzv);
            }
          }
        }
      }
    }
     auto tBelowEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tBelowElapsed = tBelowEnd - tBelowStart;
  std::cout << "[MixLength] \t cells with wall below: DONE on CPU, elapsed time: "
            << tBelowElapsed.count() << " s" << std::endl;
  }

  auto tBackStart = std::chrono::high_resolution_clock::now();

  // Apply mixing length to the cells with wall in back
  for (size_t id = 0; id < wall_back_work.size(); ++id) {
    const int id_cc = wall_back_work[id];
    const int k = static_cast<int>(id_cc / plane);
    const int j = static_cast<int>((id_cc - k * plane) / nxc);
    const int i = id_cc - j * nxc - k * plane;

    int maxdist = 0;
    if (i + k < nx - 1) {
      maxdist = k;
    } else {
      maxdist = nx - 1 - i;
    }

    WGD->mixingLengths[id_cc] = static_cast<double>(x_cc[i] - x_fc[i]);

    const float x1 = x_fc[i];
    const float y1 = y_cc[j];

    for (int ii = 0; ii <= maxdist; ++ii) {
      const int iiAbs = i + ii;
      const float dxv = x_cc[iiAbs] - x1;

      const int j1 = std::max(j - ii, 0);
      const int j2 = std::min(j + ii + 1, ny - 2);

      for (int jj = j1; jj <= j2; ++jj) {
        const int idDst = iiAbs + jj * nxc + k * plane;
        const float dyv = y_cc[jj] - y1;
        updateMinDistance2D(WGD->mixingLengths[idDst], dxv, dyv);
      }
    }
  }
  std::cout << "[MixLength] \t cells with wall in the  back: DONE " << std::endl;

  auto tFrontStart = std::chrono::high_resolution_clock::now();

  // Apply mixing length to the cells with wall in front
  for (size_t id = 0; id < wall_front_work.size(); ++id) {
    const int id_cc = wall_front_work[id];
    const int k = static_cast<int>(id_cc / plane);
    const int j = static_cast<int>((id_cc - k * plane) / nxc);
    const int i = id_cc - j * nxc - k * plane;

    int maxdist = 0;
    if (i - k > 0) {
      maxdist = k;
    } else {
      maxdist = i;
    }

    WGD->mixingLengths[id_cc] = static_cast<double>(x_fc[i + 1] - x_cc[i]);

    const float x1 = x_fc[i + 1];
    const float y1 = y_cc[j];

    for (int ii = 0; ii >= -maxdist; --ii) {
      const int iiAbs = i + 1 + ii;
      const float dxv = x_cc[iiAbs] - x1;

      const int j1 = std::max(j + ii, 0);
      const int j2 = std::min(j - ii + 1, ny - 2);

      for (int jj = j1; jj <= j2; ++jj) {
        const int idDst = iiAbs + jj * nxc + k * plane;
        const float dyv = y_cc[jj] - y1;
        updateMinDistance2D(WGD->mixingLengths[idDst], dxv, dyv);
      }
    }
  }
  std::cout << "[MixLength] \t cells with wall in the front: DONE " << std::endl;

  auto tRightStart = std::chrono::high_resolution_clock::now();

  // Apply mixing length to the cells with wall to right
  for (size_t id = 0; id < wall_right_work.size(); ++id) {
    const int id_cc = wall_right_work[id];
    const int k = static_cast<int>(id_cc / plane);
    const int j = static_cast<int>((id_cc - k * plane) / nxc);
    const int i = id_cc - j * nxc - k * plane;

    int maxdist = 0;
    if (j + k < ny - 1) {
      maxdist = k;
    } else {
      maxdist = ny - 1 - j;
    }

    WGD->mixingLengths[id_cc] = static_cast<double>(y_cc[j] - y_fc[j]);

    const float x1 = x_cc[i];
    const float y1 = y_fc[j];

    for (int jj = 0; jj <= maxdist; ++jj) {
      const int jjAbs = j + jj;
      const float dyv = y_cc[jjAbs] - y1;

      const int i1 = std::max(i - jj, 0);
      const int i2 = std::min(i + jj + 1, nx - 2);

      for (int ii = i1; ii <= i2; ++ii) {
        const int idDst = ii + jjAbs * nxc + k * plane;
        const float dxv = x_cc[ii] - x1;
        updateMinDistance2D(WGD->mixingLengths[idDst], dxv, dyv);
      }
    }
  }
  std::cout << "[MixLength] \t cells with wall to the right: DONE " << std::endl;

  auto tLeftStart = std::chrono::high_resolution_clock::now();

  // Apply mixing length to the cells with wall to left
  for (size_t id = 0; id < wall_left_work.size(); ++id) {
    const int id_cc = wall_left_work[id];
    const int k = static_cast<int>(id_cc / plane);
    const int j = static_cast<int>((id_cc - k * plane) / nxc);
    const int i = id_cc - j * nxc - k * plane;

    int maxdist = 0;
    if (j - k > 0) {
      maxdist = k;
    } else {
      maxdist = j;
    }

    WGD->mixingLengths[id_cc] = static_cast<double>(y_fc[j + 1] - y_cc[j]);

    const float x1 = x_cc[i];
    const float y1 = y_fc[j + 1];

    for (int jj = 0; jj >= -maxdist; --jj) {
      const int jjAbs = j + 1 + jj;
      const float dyv = y_cc[jjAbs] - y1;

      const int i1 = std::max(i + jj, 0);
      const int i2 = std::min(i - jj + 1, nx - 2);

      for (int ii = i1; ii <= i2; ++ii) {
        const int idDst = ii + jjAbs * nxc + k * plane;
        const float dxv = x_cc[ii] - x1;
        updateMinDistance2D(WGD->mixingLengths[idDst], dxv, dyv);
      }
    }
  }
  std::cout << "[MixLength] \t cells with wall to the left: DONE " << std::endl;
}
