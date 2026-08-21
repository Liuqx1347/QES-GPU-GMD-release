#include "LocalMixingSerialGPU.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "WINDSGeneralData.h"

namespace {

#define LM_CUDA_CHECK(call)                                                     \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::cerr << "[LocalMixingGPU] CUDA error at " << __FILE__ << ":"       \
                << __LINE__ << " -> " << cudaGetErrorString(err__)            \
                << std::endl;                                                   \
      return false;                                                             \
    }                                                                           \
  } while (0)

struct BelowSourceGPU {
  int i;
  int j;
  int k;
  int order;          // 在 CPU 原始 wall_below_indices 中的顺序
  int vertical_only;  // 1 = 四周都是 building corner，只需要竖直传播
  float x1;
  float y1;
  float z1;
};

__global__ void wallBelowGatherKernel(
    const int *target_ids,
    int n_targets,
    const int *icellflag,
    const double *mixing_in,
    double *mixing_out,
    const int *reset_order,
    const double *reset_value,
    int nx,
    int ny,
    int nz,
    int max_height,
    const float *x_cc,
    const float *y_cc,
    const float *z_cc,
    const BelowSourceGPU *sources,
    int n_sources)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_targets) return;

  const int nxc = nx - 1;
  const int plane = (nx - 1) * (ny - 1);
  const int id_cc = target_ids[tid];

  if (icellflag[id_cc] == 0 || icellflag[id_cc] == 2) {
    return;
  }

  const int k = id_cc / plane;
  const int j = (id_cc - k * plane) / nxc;
  const int i = id_cc - j * nxc - k * plane;

  const int myResetOrder = reset_order[id_cc];

  // 默认从 CPU 传入的 mixingLengths 初值开始。
  // 如果当前 target cell 本身也是 wall-below cell，则 CPU 原始串行代码处理到它时，
  // 会先执行 mixingLengths[id_cc] = z_cc[k] - z_fc[k]，覆盖更早 source 的贡献。
  // 这里用 reset_order / reset_value 显式模拟这个顺序相关行为。
  double current = mixing_in[id_cc];
  if (myResetOrder >= 0) {
    current = reset_value[id_cc];
  }

  const float xt = x_cc[i];
  const float yt = y_cc[j];
  const float zt = z_cc[k];

  for (int s = 0; s < n_sources; ++s) {
    const BelowSourceGPU src = sources[s];

    // 如果 target cell 本身会被 CPU reset，那么比它更早的 source 贡献会被覆盖掉，
    // 因此 GPU 中必须跳过这些更早 source，才能和 CPU 原始串行顺序对齐。
    if (myResetOrder >= 0 && src.order < myResetOrder) {
      continue;
    }

    if (k < src.k || k > max_height) continue;

    if (src.vertical_only) {
      if (i == src.i && j == src.j) {
        const double cand = static_cast<double>(zt - src.z1);
        if (cand < current) current = cand;
      }
      continue;
    }

    const int kk = k - src.k;
    const int reach = (max_height - src.k) + kk;
    const int di = i - src.i;
    const int dj = j - src.j;
    if ((di < 0 ? -di : di) > reach || (dj < 0 ? -dj : dj) > reach) continue;

    const double dx = static_cast<double>(xt - src.x1);
    const double dy = static_cast<double>(yt - src.y1);
    const double dz = static_cast<double>(zt - src.z1);

    const double adx = dx < 0.0 ? -dx : dx;
    const double ady = dy < 0.0 ? -dy : dy;
    const double adz = dz < 0.0 ? -dz : dz;
    if (adx >= current || ady >= current || adz >= current) {
      continue;
    }

    const double cand2 = dx * dx + dy * dy + dz * dz;
    if (cand2 < current * current) {
      current = sqrt(cand2);
    }
  }

  mixing_out[id_cc] = current;
}

} // namespace

bool localMixingWallBelowGPU(
    WINDSGeneralData *WGD,
    int max_height,
    const std::vector<float> &x_cc,
    const std::vector<float> &y_cc,
    const std::vector<float> &z_cc,
    const std::vector<float> &z_fc,
    const std::vector<int> &wall_below_work)
{
  if (!WGD) return false;
  if (wall_below_work.empty()) return true;

  const int nx = WGD->nx;
  const int ny = WGD->ny;
  const int nz = WGD->nz;
  const int nxc = nx - 1;
  const int nyc = ny - 1;
  const int plane = nxc * nyc;
  const int n_cells = plane * (nz - 1);

  if ((int)x_cc.size() < nxc ||
      (int)y_cc.size() < nyc ||
      (int)z_cc.size() < nz - 1 ||
      (int)z_fc.size() < nz) {
    std::cerr << "[LocalMixingGPU] coordinate vector size mismatch" << std::endl;
    return false;
  }

  if ((int)WGD->icellflag.size() < n_cells ||
      (int)WGD->mixingLengths.size() < n_cells) {
    std::cerr << "[LocalMixingGPU] field size mismatch" << std::endl;
    return false;
  }

  // --------------------------------------------------------------------------
  // 1. 先构造 reset_order / reset_value。
  //    这一步必须放在构造 sources 之前，因为 sources 里面要用 reset_order[id_cc]。
  //    循环顺序必须和 CPU 原始 LocalMixingSerial.cpp 一致：i -> j -> k。
  // --------------------------------------------------------------------------
  std::vector<int> reset_order(n_cells, -1);
  std::vector<double> reset_value(n_cells, 0.0);

  int wallBelowOrder = 0;
  for (int i = 1; i < nx - 2; ++i) {
    for (int j = 1; j < ny - 2; ++j) {
      for (int k = 1; k < nz - 2; ++k) {
        const int id_cc = i + j * nxc + k * plane;

        if (WGD->icellflag[id_cc] == 0 || WGD->icellflag[id_cc] == 2) {
          continue;
        }

        if (WGD->icellflag[id_cc - plane] == 0 ||
            WGD->icellflag[id_cc - plane] == 2) {
          reset_order[id_cc] = wallBelowOrder;
          reset_value[id_cc] = static_cast<double>(z_cc[k] - z_fc[k]);
          ++wallBelowOrder;
        }
      }
    }
  }

  // --------------------------------------------------------------------------
  // 2. 构造真正送入 GPU 做传播计算的 wall-below sources。
  //    wall_below_work 只包含非 terrain-only 的复杂 wall-below cell。
  // --------------------------------------------------------------------------
  std::vector<BelowSourceGPU> sources;
  sources.reserve(wall_below_work.size());

  for (size_t idx = 0; idx < wall_below_work.size(); ++idx) {
    const int id_cc = wall_below_work[idx];
    const int k = id_cc / plane;
    const int j = (id_cc - k * plane) / nxc;
    const int i = id_cc - j * nxc - k * plane;

    const int idxp = id_cc - plane + 1;
    const int idxm = id_cc - plane - 1;
    const int idyp = id_cc - plane + nxc;
    const int idym = id_cc - plane - nxc;

    BelowSourceGPU src;
    src.i = i;
    src.j = j;
    src.k = k;
    src.order = reset_order[id_cc];
    src.vertical_only =
        (WGD->icellflag[idxp] == 0 &&
         WGD->icellflag[idxm] == 0 &&
         WGD->icellflag[idyp] == 0 &&
         WGD->icellflag[idym] == 0) ? 1 : 0;
    src.x1 = x_cc[i];
    src.y1 = y_cc[j];
    src.z1 = z_fc[k];
    sources.push_back(src);
  }

  // --------------------------------------------------------------------------
  // 3. 构造 target cells。
  //    只对 max_height 范围内的 active/fluid cell 做 gather 更新。
  // --------------------------------------------------------------------------
  std::vector<int> target_ids;
  target_ids.reserve(static_cast<size_t>(nxc) *
                     static_cast<size_t>(nyc) *
                     static_cast<size_t>(std::max(1, max_height)));

  for (int k = 1; k <= max_height && k < nz - 1; ++k) {
    for (int j = 1; j < ny - 2; ++j) {
      for (int i = 1; i < nx - 2; ++i) {
        const int id_cc = i + j * nxc + k * plane;
        if (WGD->icellflag[id_cc] != 0 && WGD->icellflag[id_cc] != 2) {
          target_ids.push_back(id_cc);
        }
      }
    }
  }

  if (target_ids.empty()) return true;
  if (sources.empty()) return true;

  const int n_targets = static_cast<int>(target_ids.size());
  const int n_sources = static_cast<int>(sources.size());

  int *d_target_ids = nullptr;
  int *d_icellflag = nullptr;
  int *d_reset_order = nullptr;
  float *d_x_cc = nullptr;
  float *d_y_cc = nullptr;
  float *d_z_cc = nullptr;
  BelowSourceGPU *d_sources = nullptr;
  double *d_mixing_in = nullptr;
  double *d_mixing_out = nullptr;
  double *d_reset_value = nullptr;

  LM_CUDA_CHECK(cudaMalloc((void **)&d_target_ids, n_targets * sizeof(int)));
  LM_CUDA_CHECK(cudaMalloc((void **)&d_icellflag, n_cells * sizeof(int)));
  LM_CUDA_CHECK(cudaMalloc((void **)&d_reset_order, n_cells * sizeof(int)));
  LM_CUDA_CHECK(cudaMalloc((void **)&d_x_cc, nxc * sizeof(float)));
  LM_CUDA_CHECK(cudaMalloc((void **)&d_y_cc, nyc * sizeof(float)));
  LM_CUDA_CHECK(cudaMalloc((void **)&d_z_cc, (nz - 1) * sizeof(float)));
  LM_CUDA_CHECK(cudaMalloc((void **)&d_sources, n_sources * sizeof(BelowSourceGPU)));
  LM_CUDA_CHECK(cudaMalloc((void **)&d_mixing_in, n_cells * sizeof(double)));
  LM_CUDA_CHECK(cudaMalloc((void **)&d_mixing_out, n_cells * sizeof(double)));
  LM_CUDA_CHECK(cudaMalloc((void **)&d_reset_value, n_cells * sizeof(double)));

  LM_CUDA_CHECK(cudaMemcpy(d_target_ids, target_ids.data(),
                           n_targets * sizeof(int), cudaMemcpyHostToDevice));
  LM_CUDA_CHECK(cudaMemcpy(d_icellflag, WGD->icellflag.data(),
                           n_cells * sizeof(int), cudaMemcpyHostToDevice));
  LM_CUDA_CHECK(cudaMemcpy(d_reset_order, reset_order.data(),
                           n_cells * sizeof(int), cudaMemcpyHostToDevice));
  LM_CUDA_CHECK(cudaMemcpy(d_x_cc, x_cc.data(),
                           nxc * sizeof(float), cudaMemcpyHostToDevice));
  LM_CUDA_CHECK(cudaMemcpy(d_y_cc, y_cc.data(),
                           nyc * sizeof(float), cudaMemcpyHostToDevice));
  LM_CUDA_CHECK(cudaMemcpy(d_z_cc, z_cc.data(),
                           (nz - 1) * sizeof(float), cudaMemcpyHostToDevice));
  LM_CUDA_CHECK(cudaMemcpy(d_sources, sources.data(),
                           n_sources * sizeof(BelowSourceGPU), cudaMemcpyHostToDevice));
  LM_CUDA_CHECK(cudaMemcpy(d_mixing_in, WGD->mixingLengths.data(),
                           n_cells * sizeof(double), cudaMemcpyHostToDevice));
  LM_CUDA_CHECK(cudaMemcpy(d_mixing_out, WGD->mixingLengths.data(),
                           n_cells * sizeof(double), cudaMemcpyHostToDevice));
  LM_CUDA_CHECK(cudaMemcpy(d_reset_value, reset_value.data(),
                           n_cells * sizeof(double), cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (n_targets + threads - 1) / threads;

  wallBelowGatherKernel<<<blocks, threads>>>(
      d_target_ids,
      n_targets,
      d_icellflag,
      d_mixing_in,
      d_mixing_out,
      d_reset_order,
      d_reset_value,
      nx,
      ny,
      nz,
      max_height,
      d_x_cc,
      d_y_cc,
      d_z_cc,
      d_sources,
      n_sources);

  LM_CUDA_CHECK(cudaPeekAtLastError());
  LM_CUDA_CHECK(cudaDeviceSynchronize());

  LM_CUDA_CHECK(cudaMemcpy(WGD->mixingLengths.data(), d_mixing_out,
                           n_cells * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(d_target_ids);
  cudaFree(d_icellflag);
  cudaFree(d_reset_order);
  cudaFree(d_x_cc);
  cudaFree(d_y_cc);
  cudaFree(d_z_cc);
  cudaFree(d_sources);
  cudaFree(d_mixing_in);
  cudaFree(d_mixing_out);
  cudaFree(d_reset_value);

  return true;
}
