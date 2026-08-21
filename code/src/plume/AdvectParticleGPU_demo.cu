#include "AdvectParticleGPU_demo.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <vector>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include "Particle.hpp"
#include "Deposition.h"
#include "winds/WINDSGeneralData.h"
#include "winds/TURBGeneralData.h"

static inline void cudaCheck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::ostringstream oss;
    oss << "[CUDA ERROR] " << what << ": " << cudaGetErrorString(e);
    throw std::runtime_error(oss.str());
  }
}

__global__ void init_rng_states_kernel(curandStatePhilox4_32_10_t* states, int n, unsigned long long seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    curand_init(seed, i, 0ULL, &states[i]);
  }
}

__global__ void init_rng_states_range_kernel(
    curandStatePhilox4_32_10_t* states,
    int begin,
    int end,
    unsigned long long seed
) {
  int i = begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i < end) {
    curand_init(seed, i, 0ULL, &states[i]);
  }
}

__device__ __forceinline__
int gpu_clamp_int(int v, int lo, int hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

__device__ __forceinline__
int gpu_lower_bound_double(const double* arr, int n, double x) {
  int left = 0;
  int right = n;
  while (left < right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] < x) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

__device__ __forceinline__
double gpu_calcCourantTimestep_wall(
    const double d,
    const double u,
    const double v,
    const double w,
    const double timeRemainder,
    const double courantNum,
    const double dxy,
    const double dz,
    const double sim_dt
) {
  if (courantNum == 0.0) {
    return timeRemainder;
  }

  double min_ds = fmin(dxy, dz);
  double max_u = sqrt(u * u + v * v + w * w);

  if (max_u < 1.0e-12) {
    return timeRemainder;
  }

  double CN = 0.0;

  if (d > 6.0 * max_u * sim_dt) {
    return timeRemainder;
  } else if (d > 3.0 * max_u * sim_dt) {
    CN = fmin(2.0 * courantNum, 1.0);
  } else {
    CN = courantNum;
  }

  double dt_par = CN * min_ds / max_u;
  return fmin(timeRemainder, dt_par);
}

__device__ __forceinline__
void gpu_calcInvariants(
    const double txx,
    const double txy,
    const double txz,
    const double tyy,
    const double tyz,
    const double tzz,
    double &invar_xx,
    double &invar_yy,
    double &invar_zz
) {
  invar_xx = txx + tyy + tzz;
  invar_yy = txx * tyy + txx * tzz + tyy * tzz - txy * txy - txz * txz - tyz * tyz;
  invar_zz = txx * (tyy * tzz - tyz * tyz)
           - txy * (txy * tzz - tyz * txz)
           + txz * (txy * tyz - tyy * txz);
}

__device__ __forceinline__
bool gpu_invert3(
    double &A_11,
    double &A_12,
    double &A_13,
    double &A_21,
    double &A_22,
    double &A_23,
    double &A_31,
    double &A_32,
    double &A_33
) {
  double det = A_11 * (A_22 * A_33 - A_23 * A_32)
             - A_12 * (A_21 * A_33 - A_23 * A_31)
             + A_13 * (A_21 * A_32 - A_22 * A_31);

  if (fabs(det) < 1.0e-10) {
    A_11 = A_12 = A_13 = 0.0;
    A_21 = A_22 = A_23 = 0.0;
    A_31 = A_32 = A_33 = 0.0;
    return false;
  }

  double Ainv_11 =  (A_22 * A_33 - A_23 * A_32) / det;
  double Ainv_12 = -(A_12 * A_33 - A_13 * A_32) / det;
  double Ainv_13 =  (A_12 * A_23 - A_22 * A_13) / det;

  double Ainv_21 = -(A_21 * A_33 - A_23 * A_31) / det;
  double Ainv_22 =  (A_11 * A_33 - A_13 * A_31) / det;
  double Ainv_23 = -(A_11 * A_23 - A_13 * A_21) / det;

  double Ainv_31 =  (A_21 * A_32 - A_31 * A_22) / det;
  double Ainv_32 = -(A_11 * A_32 - A_12 * A_31) / det;
  double Ainv_33 =  (A_11 * A_22 - A_12 * A_21) / det;

  A_11 = Ainv_11; A_12 = Ainv_12; A_13 = Ainv_13;
  A_21 = Ainv_21; A_22 = Ainv_22; A_23 = Ainv_23;
  A_31 = Ainv_31; A_32 = Ainv_32; A_33 = Ainv_33;

  return true;
}

__device__ __forceinline__
void gpu_matmult(
    const double &A_11,
    const double &A_12,
    const double &A_13,
    const double &A_21,
    const double &A_22,
    const double &A_23,
    const double &A_31,
    const double &A_32,
    const double &A_33,
    const double &b_11,
    const double &b_21,
    const double &b_31,
    double &x_11,
    double &x_21,
    double &x_31
) {
  x_11 = b_11 * A_11 + b_21 * A_12 + b_31 * A_13;
  x_21 = b_11 * A_21 + b_21 * A_22 + b_31 * A_23;
  x_31 = b_11 * A_31 + b_21 * A_32 + b_31 * A_33;
}

__device__ __forceinline__
void gpu_makeRealizable(
    double &txx,
    double &txy,
    double &txz,
    double &tyy,
    double &tyz,
    double &tzz,
    const double invarianceTol
) {
  double invar_xx = 0.0;
  double invar_yy = 0.0;
  double invar_zz = 0.0;

  gpu_calcInvariants(txx, txy, txz, tyy, tyz, tzz, invar_xx, invar_yy, invar_zz);

  if (invar_xx > invarianceTol && invar_yy > invarianceTol && invar_zz > invarianceTol) {
    return;
  }

  double b = 4.0 / 3.0 * (txx + tyy + tzz);
  double c = txx * tyy + txx * tzz + tyy * tzz - txy * txy - txz * txz - tyz * tyz;
  double disc = b * b - 16.0 / 3.0 * c;
  if (disc < 0.0) disc = 0.0;

  double ks = 1.01 * (-b + sqrt(disc)) / (8.0 / 3.0);
  if (ks < invarianceTol || isnan(ks)) {
    ks = 0.5 * fabs(txx + tyy + tzz);
  }

  double txx_new = txx + 2.0 / 3.0 * ks;
  double txy_new = txy;
  double txz_new = txz;
  double tyy_new = tyy + 2.0 / 3.0 * ks;
  double tyz_new = tyz;
  double tzz_new = tzz + 2.0 / 3.0 * ks;

  gpu_calcInvariants(txx_new, txy_new, txz_new, tyy_new, tyz_new, tzz_new, invar_xx, invar_yy, invar_zz);

  int iter = 0;
  while ((invar_xx < invarianceTol || invar_yy < invarianceTol || invar_zz < invarianceTol) && iter < 1000) {
    iter++;
    ks *= 1.05;

    txx_new = txx + 2.0 / 3.0 * ks;
    tyy_new = tyy + 2.0 / 3.0 * ks;
    tzz_new = tzz + 2.0 / 3.0 * ks;

    gpu_calcInvariants(txx_new, txy_new, txz_new, tyy_new, tyz_new, tzz_new, invar_xx, invar_yy, invar_zz);
  }

  txx = txx_new;
  txy = txy_new;
  txz = txz_new;
  tyy = tyy_new;
  tyz = tyz_new;
  tzz = tzz_new;
}

__device__ __forceinline__
bool gpu_enforceBC_exiting(double &pos, const double domainStart, const double domainEnd) {
  if (pos <= domainStart || pos >= domainEnd) {
    return false;
  } else {
    return true;
  }
}

__device__ __forceinline__
bool gpu_inside_physical_domain(
    const double xPos,
    const double yPos,
    const double zPos,
    const double xStart,
    const double xEnd,
    const double yStart,
    const double yEnd,
    const double zStart,
    const double zEnd
) {
  const double eps = 1.0e-10;
  return (xPos > xStart + eps && xPos < xEnd - eps &&
          yPos > yStart + eps && yPos < yEnd - eps &&
          zPos > zStart + eps && zPos < zEnd - eps);
}

__device__ __forceinline__
bool gpu_inside_bc_domain(
    const double xPos,
    const double yPos,
    const double zPos,
    const double xBCStart,
    const double xBCEnd,
    const double yBCStart,
    const double yBCEnd,
    const double zBCStart,
    const double zBCEnd
) {
  const double eps = 1.0e-10;
  return (xPos > xBCStart + eps && xPos < xBCEnd - eps &&
          yPos > yBCStart + eps && yPos < yBCEnd - eps &&
          zPos > zBCStart + eps && zPos < zBCEnd - eps);
}

enum GpuExitReason : unsigned char {
  GPU_EXIT_NONE = 0,
  GPU_EXIT_START_OUTSIDE_PHYS = 1,
  GPU_EXIT_INV_SIGMA = 2,
  GPU_EXIT_INV_A = 3,
  GPU_EXIT_ROGUE_U = 4,
  GPU_EXIT_ROGUE_V = 5,
  GPU_EXIT_ROGUE_W = 6,
  GPU_EXIT_REFLECT_FAIL = 7,
  GPU_EXIT_BOTTOM_CLAMP = 8,
  GPU_EXIT_X_BC = 9,
  GPU_EXIT_Y_BC = 10,
  GPU_EXIT_Z_BC = 11,
  GPU_EXIT_END_OUTSIDE_PHYS = 12,
  GPU_EXIT_END_INVALID_CELL = 13,
  GPU_EXIT_MAX_SUBSTEPS = 14,
  GPU_EXIT_REFLECT_END_INVALID = 15,
  GPU_EXIT_REFLECT_TRIAL_INVALID = 16,
  GPU_EXIT_REFLECT_SURFACE_FAIL = 17,
  GPU_EXIT_REFLECT_MAXCOUNT = 18
};

static const int GPU_EXIT_CODE_MAX = 18;

enum GpuDomainBCType : int {
  GPU_DOMAIN_BC_EXITING   = 0,
  GPU_DOMAIN_BC_PERIODIC  = 1,
  GPU_DOMAIN_BC_REFLECT   = 2
};

__device__ __forceinline__
bool gpu_enforce_domainBC_1d(
    double &pos,
    double &velFluct,
    const double domainStart,
    const double domainEnd,
    const int bcType
) {
  if (bcType == GPU_DOMAIN_BC_EXITING) {
    // 完全对齐 CPU: DomainBC_exiting::enforce
    if (pos <= domainStart || pos >= domainEnd) {
      return false;
    } else {
      return true;
    }
  }

  if (bcType == GPU_DOMAIN_BC_PERIODIC) {
    // 完全对齐 CPU: DomainBC_periodic::enforce
    const double domainSize = domainEnd - domainStart;
    if (domainSize != 0.0) {
      while (pos < domainStart) {
        pos = pos + domainSize;
      }
      while (pos > domainEnd) {
        pos = pos - domainSize;
      }
    }
    return true;
  }

  if (bcType == GPU_DOMAIN_BC_REFLECT) {
    // 完全对齐 CPU: DomainBC_reflection::enforce
    int reflectCount = 0;
    while ((pos < domainStart || pos > domainEnd) && reflectCount < 100) {
      if (pos > domainEnd) {
        pos = domainEnd - (pos - domainEnd);
        velFluct = -velFluct;
      } else if (pos < domainStart) {
        pos = domainStart - (pos - domainStart);
        velFluct = -velFluct;
      }
      reflectCount = reflectCount + 1;
    }

    if (reflectCount == 100) {
      if (pos > domainEnd) {
        return false;
      } else if (pos < domainStart) {
        return false;
      }
    }

    return true;
  }

  // 兜底：未知类型时按 exiting 处理
  if (pos <= domainStart || pos >= domainEnd) {
    return false;
  }
  return true;
}

__device__ __forceinline__
void gpu_set_exit_reason(unsigned char &reason, const unsigned char code) {
  if (reason == GPU_EXIT_NONE) {
    reason = code;
  }
}

static const char* gpu_exit_reason_name(const unsigned char code) {
  switch (code) {
    case GPU_EXIT_NONE: return "still-active/no-exit";
    case GPU_EXIT_START_OUTSIDE_PHYS: return "start-outside-physical-domain";
    case GPU_EXIT_INV_SIGMA: return "invert-sigma-failed";
    case GPU_EXIT_INV_A: return "invert-A-failed";
    case GPU_EXIT_ROGUE_U: return "rogue-uFluct";
    case GPU_EXIT_ROGUE_V: return "rogue-vFluct";
    case GPU_EXIT_ROGUE_W: return "rogue-wFluct";
    case GPU_EXIT_REFLECT_FAIL: return "reflection-failed";
    case GPU_EXIT_BOTTOM_CLAMP: return "bottom-clamp/deactivate";
    case GPU_EXIT_X_BC: return "exit-x-boundary";
    case GPU_EXIT_Y_BC: return "exit-y-boundary";
    case GPU_EXIT_Z_BC: return "exit-z-boundary";
    case GPU_EXIT_END_OUTSIDE_PHYS: return "end-outside-physical-domain";
    case GPU_EXIT_END_INVALID_CELL: return "end-invalid/solid-cell";
    case GPU_EXIT_MAX_SUBSTEPS: return "max-substeps-reached";
    case GPU_EXIT_REFLECT_END_INVALID: return "reflect-end-invalid";
    case GPU_EXIT_REFLECT_TRIAL_INVALID: return "reflect-trial-invalid";
    case GPU_EXIT_REFLECT_SURFACE_FAIL: return "reflect-surface-fail";
    case GPU_EXIT_REFLECT_MAXCOUNT: return "reflect-maxcount";
    default: return "unknown";
  }
}


__device__ __forceinline__
bool gpu_tryGetCellId(
    const double xPos,
    const double yPos,
    const double zPos,
    const double* zFaces,
    const int nx,
    const int ny,
    const int nz,
    const double dx,
    const double dy,
    const double dz,
    int &cellId
) {
  (void)dz;
  int i = (int)floor((xPos - 0.0 * dx) / (dx + 1e-9));
  int j = (int)floor((yPos - 0.0 * dy) / (dy + 1e-9));
  int k = gpu_lower_bound_double(zFaces, nz, zPos) - 1;

  if (i < 0 || i > nx - 2 ||
      j < 0 || j > ny - 2 ||
      k < 0 || k > nz - 2) {
    cellId = -1;
    return false;
  }

  cellId = i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
  return true;
}

__device__ __forceinline__
void gpu_getCellIndex_fromCellId(
    const int cellId,
    const int nx,
    const int ny,
    int &i,
    int &j,
    int &k
) {
  k = (int)(cellId / ((nx - 1) * (ny - 1)));
  j = (int)((cellId - k * (nx - 1) * (ny - 1)) / (nx - 1));
  i = cellId - j * (nx - 1) - k * (nx - 1) * (ny - 1);
}


struct GpuVec3d {
  double x, y, z;
  __device__ __forceinline__ GpuVec3d() : x(0.0), y(0.0), z(0.0) {}
  __device__ __forceinline__ GpuVec3d(double ax, double ay, double az) : x(ax), y(ay), z(az) {}
  __device__ __forceinline__ GpuVec3d operator+(const GpuVec3d& b) const { return GpuVec3d(x + b.x, y + b.y, z + b.z); }
  __device__ __forceinline__ GpuVec3d operator-(const GpuVec3d& b) const { return GpuVec3d(x - b.x, y - b.y, z - b.z); }
  __device__ __forceinline__ GpuVec3d operator*(double a) const { return GpuVec3d(x * a, y * a, z * a); }
  __device__ __forceinline__ GpuVec3d operator/(double a) const { return GpuVec3d(x / a, y / a, z / a); }
  __device__ __forceinline__ GpuVec3d& operator+=(const GpuVec3d& b) { x+=b.x; y+=b.y; z+=b.z; return *this; }
};
__device__ __forceinline__ GpuVec3d operator*(double a, const GpuVec3d& v) { return GpuVec3d(a*v.x, a*v.y, a*v.z); }
__device__ __forceinline__ double gpu_dot(const GpuVec3d& a, const GpuVec3d& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ __forceinline__ double gpu_len(const GpuVec3d& a) { return sqrt(a.x*a.x + a.y*a.y + a.z*a.z); }
__device__ __forceinline__ GpuVec3d gpu_normalize(const GpuVec3d& a) { double l = gpu_len(a); return (l < 1e-12) ? GpuVec3d(0,0,0) : a / l; }
__device__ __forceinline__ GpuVec3d gpu_reflect_vec(const GpuVec3d& v, const GpuVec3d& n) { return v - 2.0 * gpu_dot(v, n) * n; }

__device__ __forceinline__
bool gpu_cellflag_at(
    const int* icellflag,
    const double* zFaces,
    const int nx,
    const int ny,
    const int nz,
    const double dx,
    const double dy,
    const double dz,
    const double domainZstart,
    const GpuVec3d& X,
    int& cellId,
    int& cellFlag
) {
  bool inRange = gpu_tryGetCellId(X.x, X.y, X.z, zFaces, nx, ny, nz, dx, dy, dz, cellId);
  if (!inRange) {
    if (X.z < domainZstart) {
      cellFlag = 2;
      cellId = -1;
      return true;
    }
    cellFlag = -999;
    cellId = -1;
    return false;
  }
  cellFlag = icellflag[cellId];
  return true;
}

__device__ __forceinline__
bool gpu_oneReflection_full(
    const int* icellflag,
    const double* xCenters,
    const double* yCenters,
    const double* zFaces,
    const int nx,
    const int ny,
    const int nz,
    const double dx,
    const double dy,
    const double dz,
    const double domainZstart,
    GpuVec3d &X,
    GpuVec3d &u,
    const double d,
    GpuVec3d &vecFluct,
    bool &isActive,
    unsigned char &exitReason
) {
  const double eps_S = 1.0e-3;
  const int maxCount = 10;
  const GpuVec3d e1(1.0,0.0,0.0), e2(0.0,1.0,0.0), e3(0.0,0.0,1.0);

  GpuVec3d Xold = X;
  GpuVec3d U = d * u;
  GpuVec3d Xnew = Xold + U;

  int cellIdNew, cellFlagNew;
  if (!gpu_cellflag_at(icellflag, zFaces, nx, ny, nz, dx, dy, dz, domainZstart, Xnew, cellIdNew, cellFlagNew)) {
    gpu_set_exit_reason(exitReason, GPU_EXIT_REFLECT_END_INVALID);
    isActive = false;
    return false;
  }

  int count = 0;
  while ((cellFlagNew == 0 || cellFlagNew == 2) && (count < maxCount)) {
    int cellIdOld, cellFlagOld;
    if (!gpu_cellflag_at(icellflag, zFaces, nx, ny, nz, dx, dy, dz, domainZstart, Xold, cellIdOld, cellFlagOld) || cellIdOld < 0) {
      gpu_set_exit_reason(exitReason, GPU_EXIT_REFLECT_TRIAL_INVALID);
      isActive = false;
      return false;
    }

    int i,j,k;
    gpu_getCellIndex_fromCellId(cellIdOld, nx, ny, i, j, k);

    double f1 = (fabs(U.x) > 1e-12) ? (U.x > 0.0 ? 1.0 : -1.0) : 0.0;
    double f2 = (fabs(U.y) > 1e-12) ? (U.y > 0.0 ? 1.0 : -1.0) : 0.0;
    double f3 = (fabs(U.z) > 1e-12) ? (U.z > 0.0 ? 1.0 : -1.0) : 0.0;

    double l1 = 1e30, l2 = 1e30, l3 = 1e30;
    GpuVec3d N1, N2, N3;
    if (f1 != 0.0) {
      N1 = (-f1) * e1;
      GpuVec3d Sx(xCenters[i] + f1 * 0.5 * dx, yCenters[j], 0.5 * (zFaces[k] + zFaces[k+1]));
      double denom = gpu_dot(U, N1);
      if (fabs(denom) > 1e-12) l1 = -(gpu_dot(Xold, N1) - gpu_dot(Sx, N1)) / denom;
    }
    if (f2 != 0.0) {
      N2 = (-f2) * e2;
      GpuVec3d Sy(xCenters[i], yCenters[j] + f2 * 0.5 * dy, 0.5 * (zFaces[k] + zFaces[k+1]));
      double denom = gpu_dot(U, N2);
      if (fabs(denom) > 1e-12) l2 = -(gpu_dot(Xold, N2) - gpu_dot(Sy, N2)) / denom;
    }
    if (f3 != 0.0) {
      N3 = (-f3) * e3;
      GpuVec3d Sz(xCenters[i], yCenters[j], (f3 >= 0.0) ? zFaces[k+1] : zFaces[k]);
      double denom = gpu_dot(U, N3);
      if (fabs(denom) > 1e-12) l3 = -(gpu_dot(Xold, N3) - gpu_dot(Sz, N3)) / denom;
    }

    int validSurface = 0;
    double s = 100.0;
    GpuVec3d N;

    if ((l1 >= -eps_S) && (l1 <= 1.0 - eps_S)) {
      validSurface++;
    } else if ((l1 >= 1.0 - eps_S) && (l1 <= 1.0 + eps_S)) {
      validSurface++;
      l1 -= 2.0 * eps_S;
    }
    if ((l2 >= -eps_S) && (l2 <= 1.0 - eps_S)) {
      validSurface++;
    } else if ((l2 >= 1.0 - eps_S) && (l2 <= 1.0 + eps_S)) {
      validSurface++;
      l2 -= 2.0 * eps_S;
    }
    if ((l3 >= -eps_S) && (l3 <= 1.0 - eps_S)) {
      validSurface++;
    } else if ((l3 >= 1.0 - eps_S) && (l3 <= 1.0 + eps_S)) {
      validSurface++;
      l3 -= 2.0 * eps_S;
    }

    if (validSurface == 0) {
      gpu_set_exit_reason(exitReason, GPU_EXIT_REFLECT_SURFACE_FAIL);
      isActive = false;
      return false;
    } else if (validSurface == 1) {
      if (l1 <= 1.0 - eps_S && l1 >= -eps_S) { s = l1; N = N1; }
      if (l2 <= 1.0 - eps_S && l2 >= -eps_S) { s = l2; N = N2; }
      if (l3 <= 1.0 - eps_S && l3 >= -eps_S) { s = l3; N = N3; }
    } else {
      // choose the earliest valid surface that keeps the next crossed cell solid
      double vl[3] = {l1, l2, l3};
      GpuVec3d vN[3] = {N1, N2, N3};
      int vn[3] = {(int)f1, (int)(f2 * (nx - 1)), (int)(f3 * (nx - 1) * (ny - 1))};
      int order[3] = {0,1,2};
      for (int a=0;a<3;a++) for (int b=a+1;b<3;b++) if (vl[order[b]] < vl[order[a]]) { int t=order[a]; order[a]=order[b]; order[b]=t; }

      bool found = false;
      for (int oi=0; oi<3 && !found; ++oi) {
        int idx = order[oi];
        if (!(vl[idx] >= -eps_S && vl[idx] <= 1.0 + eps_S)) continue;
        int cand = cellIdOld + vn[idx];
        if (cand >= 0 && cand < (nx-1)*(ny-1)*(nz-1) && (icellflag[cand] == 0 || icellflag[cand] == 2)) {
          s = vl[idx]; N = vN[idx]; found = true; break;
        }
      }
      if (!found) {
        // fallback to smallest valid surface
        for (int oi=0; oi<3; ++oi) {
          int idx = order[oi];
          if (vl[idx] >= -eps_S && vl[idx] <= 1.0 + eps_S) { s = vl[idx]; N = vN[idx]; found = true; break; }
        }
      }
      if (!found) {
        gpu_set_exit_reason(exitReason, GPU_EXIT_REFLECT_SURFACE_FAIL);
        isActive = false;
        return false;
      }
    }

    GpuVec3d V1 = s * U;
    GpuVec3d P = Xold + V1;
    GpuVec3d V2 = U - V1;
    double r = gpu_len(V2);
    if (r < 1e-12) {
      X = P;
      isActive = true;
      return true;
    }
    GpuVec3d V2u = V2 / r;
    GpuVec3d R = gpu_reflect_vec(V2u, N);
    Xnew = P + r * R;
    vecFluct = gpu_reflect_vec(vecFluct, N);
    Xold = P;
    U = Xnew - Xold;
    count++;

    if (!gpu_cellflag_at(icellflag, zFaces, nx, ny, nz, dx, dy, dz, domainZstart, Xnew, cellIdNew, cellFlagNew)) {
      gpu_set_exit_reason(exitReason, GPU_EXIT_REFLECT_END_INVALID);
      isActive = false;
      return false;
    }
  }

  if (count < maxCount) {
    X = Xnew;
    double ul = gpu_len(U);
    u = (ul > 1e-12) ? (U / ul) : u;
    isActive = true;
    return true;
  } else {
    X = Xold;
    gpu_set_exit_reason(exitReason, GPU_EXIT_REFLECT_MAXCOUNT);
    isActive = false;
    return false;
  }
}

__device__ __forceinline__
bool gpu_reflect_stairstep_full(
    const int* icellflag,
    const double* xCenters,
    const double* yCenters,
    const double* zFaces,
    const int nx,
    const int ny,
    const int nz,
    const double dx,
    const double dy,
    const double dz,
    const double domainZstart,
    double &xPos,
    double &yPos,
    double &zPos,
    double &disX,
    double &disY,
    double &disZ,
    double &uFluct,
    double &vFluct,
    double &wFluct,
    unsigned char &exitReason
) {
    int cellIdNew, cellFlagNew;
    GpuVec3d Xend(xPos, yPos, zPos);

    if (!gpu_cellflag_at(
            icellflag, zFaces,
            nx, ny, nz, dx, dy, dz, domainZstart,
            Xend, cellIdNew, cellFlagNew)) {
        gpu_set_exit_reason(exitReason, GPU_EXIT_REFLECT_END_INVALID);
        return false;
    }

    // 不能只在“终点落入 solid/terrain”时才做 stair-step。
    // 否则会出现：粒子整段路径穿过建筑/地形，但终点又落回 fluid，GPU 版本就会直接放行，
    // 造成羽流过宽、穿墙、与 CPU 结果明显不一致。
    // 所以这里无论终点是否在 solid/terrain，都进入后续的分段检查。

    GpuVec3d Xstart(xPos - disX, yPos - disY, zPos - disZ);
    GpuVec3d X = Xstart;
    GpuVec3d vecFluct(uFluct, vFluct, wFluct);

    double dTotal = sqrt(disX * disX + disY * disY + disZ * disZ);
    if (dTotal < 1.0e-12) {
        return true;
    }

    GpuVec3d u(disX / dTotal, disY / dTotal, disZ / dTotal);

 

    double dRemain = dTotal;
    bool isActive = true;

    // 防止异常情况下死循环
    int outerGuard = 0;
    const int MAX_OUTER = 64;

    while (dRemain > 1.0e-6 && isActive && outerGuard < MAX_OUTER) {
        outerGuard++;

        // 优先尝试“剩余整段”
        double dSeg = dRemain;
        bool acceptedSeg = false;

        while (dSeg > 1.0e-6 && !acceptedSeg) {
            GpuVec3d Xtrial = X + dSeg * u;

            int cellIdOld, cellFlagOld;
            int cellIdTrial, cellFlagTrial;

            bool okOld = gpu_cellflag_at(
                icellflag, zFaces,
                nx, ny, nz, dx, dy, dz, domainZstart,
                X, cellIdOld, cellFlagOld);

            bool okTrial = gpu_cellflag_at(
                icellflag, zFaces,
                nx, ny, nz, dx, dy, dz, domainZstart,
                Xtrial, cellIdTrial, cellFlagTrial);

            if (!okOld || !okTrial) {
                gpu_set_exit_reason(exitReason, GPU_EXIT_REFLECT_TRIAL_INVALID);
                isActive = false;
                break;
            }

            if (cellIdOld < 0 || cellIdTrial < 0) {
                gpu_set_exit_reason(exitReason, GPU_EXIT_REFLECT_TRIAL_INVALID);
                isActive = false;
                break;
            }

            int i0, j0, k0;
            int i1, j1, k1;
            gpu_getCellIndex_fromCellId(cellIdOld, nx, ny, i0, j0, k0);
            gpu_getCellIndex_fromCellId(cellIdTrial, nx, ny, i1, j1, k1);

            int di = i1 - i0; if (di < 0) di = -di;
            int dj = j1 - j0; if (dj < 0) dj = -dj;
            int dk = k1 - k0; if (dk < 0) dk = -dk;

            // 如果这一整段跨单元太多，就二分
            if (di > 1 || dj > 1 || dk > 1) {
                dSeg *= 0.5;
            } else {
                acceptedSeg = true;
            }
        }

        if (!isActive) {
            return false;
        }

        // 段已经小到没法再切时，不能直接停在当前位置。
        // 旧写法会把这一小步剩余位移直接丢掉，粒子在复杂障碍附近会被“卡慢”，
        // 表现为：羽流推进过慢、活跃粒子数偏多、浓度场偏胖。
        if (dSeg <= 1.0e-6) {
            GpuVec3d Xfallback = X + dRemain * u;
            int cellIdFallback, cellFlagFallback;
            bool okFallback = gpu_cellflag_at(
                icellflag, zFaces,
                nx, ny, nz, dx, dy, dz, domainZstart,
                Xfallback, cellIdFallback, cellFlagFallback);

            if (okFallback && !(cellFlagFallback == 0 || cellFlagFallback == 2)) {
                X = Xfallback;
            }
            break;
        }

        bool ok = gpu_oneReflection_full(
            icellflag, xCenters, yCenters, zFaces,
            nx, ny, nz,
            dx, dy, dz,
            domainZstart,
            X, u, dSeg, vecFluct, isActive, exitReason
        );

        if (!ok || !isActive) {
            return false;
        }

        // 这里减掉“这一段路径长度”
        dRemain -= dSeg;
    }

    if (!isActive) {
        return false;
    }

    xPos = X.x;
    yPos = X.y;
    zPos = X.z;

    disX = xPos - Xstart.x;
    disY = yPos - Xstart.y;
    disZ = zPos - Xstart.z;

    uFluct = vecFluct.x;
    vFluct = vecFluct.y;
    wFluct = vecFluct.z;

    return true;
}

__device__ __forceinline__
int gpu_getCellId(
    const double xPos,
    const double yPos,
    const double zPos,
    const double* zFaces,
    const int nx,
    const int ny,
    const int nz,
    const double dx,
    const double dy,
    const double dz
) {
  (void)dz;
  int i = (int)floor((xPos - 0.0 * dx) / (dx + 1e-9));
  int j = (int)floor((yPos - 0.0 * dy) / (dy + 1e-9));
  int k = gpu_lower_bound_double(zFaces, nz, zPos) - 1;

  i = gpu_clamp_int(i, 0, nx - 2);
  j = gpu_clamp_int(j, 0, ny - 2);
  k = gpu_clamp_int(k, 0, nz - 2);

  return i + j * (nx - 1) + k * (nx - 1) * (ny - 1);
}

 struct GpuInterpWeight {
   int ii;
   int jj;
   int kk;
   double iw;
   double jw;
   double kw;
};

__device__ __forceinline__
void gpu_setInterp3Dindex_uFace(
    const double par_xPos,
    const double par_yPos,
    const double par_zPos,
    const double* z_nodes,
    const int nx,
    const int ny,
    const int nz,
    const double dx,
    const double dy,
    GpuInterpWeight &wgt
) {
  double par_x = par_xPos - 0.0 * dx;
  double par_y = par_yPos - 0.5 * dy;

  wgt.ii = (int)floor(par_x / (dx + 1e-7));
  wgt.jj = (int)floor(par_y / (dy + 1e-7));

  wgt.iw = (par_x / dx) - floor(par_x / (dx + 1e-7));
  wgt.jw = (par_y / dy) - floor(par_y / (dy + 1e-7));

  int idx = gpu_lower_bound_double(z_nodes, nz, par_zPos);
  wgt.kk = idx - 1;
  wgt.kk = gpu_clamp_int(wgt.kk, 0, nz - 2);

  double denom = z_nodes[wgt.kk + 1] - z_nodes[wgt.kk];
  if (fabs(denom) < 1e-12) denom = 1e-12;
  wgt.kw = (par_zPos - z_nodes[wgt.kk]) / denom;

  wgt.ii = gpu_clamp_int(wgt.ii, 0, nx - 2);
  wgt.jj = gpu_clamp_int(wgt.jj, 0, ny - 2);
}

__device__ __forceinline__
void gpu_setInterp3Dindex_vFace(
    const double par_xPos,
    const double par_yPos,
    const double par_zPos,
    const double* z_nodes,
    const int nx,
    const int ny,
    const int nz,
    const double dx,
    const double dy,
    GpuInterpWeight &wgt
) {
  double par_x = par_xPos - 0.5 * dx;
  double par_y = par_yPos - 0.0 * dy;

  wgt.ii = (int)floor(par_x / (dx + 1e-7));
  wgt.jj = (int)floor(par_y / (dy + 1e-7));

  wgt.iw = (par_x / dx) - floor(par_x / (dx + 1e-7));
  wgt.jw = (par_y / dy) - floor(par_y / (dy + 1e-7));

  int idx = gpu_lower_bound_double(z_nodes, nz, par_zPos);
  wgt.kk = idx - 1;
  wgt.kk = gpu_clamp_int(wgt.kk, 0, nz - 2);

  double denom = z_nodes[wgt.kk + 1] - z_nodes[wgt.kk];
  if (fabs(denom) < 1e-12) denom = 1e-12;
  wgt.kw = (par_zPos - z_nodes[wgt.kk]) / denom;

  wgt.ii = gpu_clamp_int(wgt.ii, 0, nx - 2);
  wgt.jj = gpu_clamp_int(wgt.jj, 0, ny - 2);
}

__device__ __forceinline__
void gpu_setInterp3Dindex_wFace(
    const double par_xPos,
    const double par_yPos,
    const double par_zPos,
    const double* z_faces,
    const int nx,
    const int ny,
    const int nz,
    const double dx,
    const double dy,
    GpuInterpWeight &wgt
) {
  double par_x = par_xPos - 0.5 * dx;
  double par_y = par_yPos - 0.5 * dy;

  wgt.ii = (int)floor(par_x / (dx + 1e-7));
  wgt.jj = (int)floor(par_y / (dy + 1e-7));

  wgt.iw = (par_x / dx) - floor(par_x / (dx + 1e-7));
  wgt.jw = (par_y / dy) - floor(par_y / (dy + 1e-7));

  int idx = gpu_lower_bound_double(z_faces, nz, par_zPos);
  wgt.kk = idx - 1;
  wgt.kk = gpu_clamp_int(wgt.kk, 0, nz - 2);

  double denom = z_faces[wgt.kk + 1] - z_faces[wgt.kk];
  if (fabs(denom) < 1e-12) denom = 1e-12;
  wgt.kw = (par_zPos - z_faces[wgt.kk]) / denom;

  wgt.ii = gpu_clamp_int(wgt.ii, 0, nx - 2);
  wgt.jj = gpu_clamp_int(wgt.jj, 0, ny - 2);
}

__device__ __forceinline__
void gpu_setInterp3Dindex_cellVar(
    const double par_xPos,
    const double par_yPos,
    const double par_zPos,
    const double* z_nodes,
    const int nx,
    const int ny,
    const int nz,
    const double dx,
    const double dy,
    GpuInterpWeight &wgt
) {
  double par_x = par_xPos - 0.5 * dx;
  double par_y = par_yPos - 0.5 * dy;

  wgt.ii = (int)floor(par_x / (dx + 1e-7));
  wgt.jj = (int)floor(par_y / (dy + 1e-7));

  wgt.iw = (par_x / dx) - floor(par_x / (dx + 1e-7));
  wgt.jw = (par_y / dy) - floor(par_y / (dy + 1e-7));

  int idx = gpu_lower_bound_double(z_nodes, nz, par_zPos);
  wgt.kk = idx - 1;
  wgt.kk = gpu_clamp_int(wgt.kk, 0, nz - 3);

  double denom = z_nodes[wgt.kk + 1] - z_nodes[wgt.kk];
  if (fabs(denom) < 1e-12) denom = 1e-12;
  wgt.kw = (par_zPos - z_nodes[wgt.kk]) / denom;

  wgt.ii = gpu_clamp_int(wgt.ii, 0, nx - 3);
  wgt.jj = gpu_clamp_int(wgt.jj, 0, ny - 3);
}

__device__ __forceinline__
void gpu_interp3D_faceVar_double(
    const double* EulerData,
    const GpuInterpWeight& wgt, 
    int nx, int ny,
    double& out
) {
  double cube[2][2][2];

  for (int kkk = 0; kkk <= 1; kkk++) {
    for (int jjj = 0; jjj <= 1; jjj++) {
      for (int iii = 0; iii <= 1; iii++) {
        int idx = (wgt.kk + kkk) * (ny * nx) + (wgt.jj + jjj) * nx + (wgt.ii + iii);
        cube[iii][jjj][kkk] = (double)EulerData[idx];
      }
    }
  }

  double u_low =
      (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][0]
    + wgt.iw       * (1 - wgt.jw) * cube[1][0][0]
    + wgt.iw       * wgt.jw       * cube[1][1][0]
    + (1 - wgt.iw) * wgt.jw       * cube[0][1][0];

  double u_high =
      (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][1]
    + wgt.iw       * (1 - wgt.jw) * cube[1][0][1]
    + wgt.iw       * wgt.jw       * cube[1][1][1]
    + (1 - wgt.iw) * wgt.jw       * cube[0][1][1];

  out = (u_high - u_low) * wgt.kw + u_low;
}

__device__ __forceinline__
void gpu_interp3D_cellVar_double(
    const double* EulerData,
    const GpuInterpWeight& wgt,
    int nx, int ny,
    double& out
) {
  double cube[2][2][2];

  for (int kkk = 0; kkk <= 1; kkk++) {
    for (int jjj = 0; jjj <= 1; jjj++) {
      for (int iii = 0; iii <= 1; iii++) {
        int idx = (wgt.kk + kkk) * (ny - 1) * (nx - 1)
                + (wgt.jj + jjj) * (nx - 1)
                + (wgt.ii + iii);
        cube[iii][jjj][kkk] = (double)EulerData[idx];
      }
    }
  }

  double u_low =
      (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][0]
    + wgt.iw       * (1 - wgt.jw) * cube[1][0][0]
    + wgt.iw       * wgt.jw       * cube[1][1][0]
    + (1 - wgt.iw) * wgt.jw       * cube[0][1][0];

  double u_high =
      (1 - wgt.iw) * (1 - wgt.jw) * cube[0][0][1]
    + wgt.iw       * (1 - wgt.jw) * cube[1][0][1]
    + wgt.iw       * wgt.jw       * cube[1][1][1]
    + (1 - wgt.iw) * wgt.jw       * cube[0][1][1];

  out = (u_high - u_low) * wgt.kw + u_low;
}

__device__ __forceinline__
void gpu_interpValuesTriLinear(
    const double xPos,
    const double yPos,
    const double zPos,
    const double* WGD_u,
    const double* WGD_v,
    const double* WGD_w,
    const double* TGD_CoEps,
    const double* TGD_txx,
    const double* TGD_txy,
    const double* TGD_txz,
    const double* TGD_tyy,
    const double* TGD_tyz,
    const double* TGD_tzz,
    const double* TGD_div_tau_x,
    const double* TGD_div_tau_y,
    const double* TGD_div_tau_z,
    const double* TGD_nuT,
    const double* z_nodes,
    const double* z_faces,
    const int nx,
    const int ny,
    const int nz,
    const double dx,
    const double dy,
    double &uMean_out,
    double &vMean_out,
    double &wMean_out,
    double &txx_out,
    double &txy_out,
    double &txz_out,
    double &tyy_out,
    double &tyz_out,
    double &tzz_out,
    double &flux_div_x_out,
    double &flux_div_y_out,
    double &flux_div_z_out,
    double &nuT_out,
    double &CoEps_out
) {
  GpuInterpWeight wgt;

  gpu_setInterp3Dindex_uFace(xPos, yPos, zPos, z_nodes, nx, ny, nz, dx, dy, wgt);
  gpu_interp3D_faceVar_double(WGD_u, wgt, nx, ny, uMean_out);

  gpu_setInterp3Dindex_vFace(xPos, yPos, zPos, z_nodes, nx, ny, nz, dx, dy, wgt);
  gpu_interp3D_faceVar_double(WGD_v, wgt, nx, ny, vMean_out);

  gpu_setInterp3Dindex_wFace(xPos, yPos, zPos, z_faces, nx, ny, nz, dx, dy, wgt);
  gpu_interp3D_faceVar_double(WGD_w, wgt, nx, ny, wMean_out);

  gpu_setInterp3Dindex_cellVar(xPos, yPos, zPos, z_nodes, nx, ny, nz, dx, dy, wgt);

  gpu_interp3D_cellVar_double(TGD_CoEps,     wgt, nx, ny, CoEps_out);
  gpu_interp3D_cellVar_double(TGD_txx,       wgt, nx, ny, txx_out);
  gpu_interp3D_cellVar_double(TGD_txy,       wgt, nx, ny, txy_out);
  gpu_interp3D_cellVar_double(TGD_txz,       wgt, nx, ny, txz_out);
  gpu_interp3D_cellVar_double(TGD_tyy,       wgt, nx, ny, tyy_out);
  gpu_interp3D_cellVar_double(TGD_tyz,       wgt, nx, ny, tyz_out);
  gpu_interp3D_cellVar_double(TGD_tzz,       wgt, nx, ny, tzz_out);

  gpu_interp3D_cellVar_double(TGD_div_tau_x, wgt, nx, ny, flux_div_x_out);
  gpu_interp3D_cellVar_double(TGD_div_tau_y, wgt, nx, ny, flux_div_y_out);
  gpu_interp3D_cellVar_double(TGD_div_tau_z, wgt, nx, ny, flux_div_z_out);

  gpu_interp3D_cellVar_double(TGD_nuT,       wgt, nx, ny, nuT_out);

  if (CoEps_out <= 1.0e-6) {
    CoEps_out = 1.0e-6;
  }
}

__device__ __forceinline__
bool gpu_isCanopyFlag(const int flag) {
  return (flag == 20 || flag == 22 || flag == 24 || flag == 28);
}

__device__ __forceinline__
bool gpu_isTerrainFlag(const int flag) {
  return (flag == 2);
}

__device__ __forceinline__
double gpu_atomicAdd_double(double* address, double val)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
  return atomicAdd(address, val);
#else
  unsigned long long int* address_as_ull =
      reinterpret_cast<unsigned long long int*>(address);

  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull,
        assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
#endif
}

__device__ __forceinline__
void gpu_apply_deposition(
    double xPos, double yPos, double zPos,
    double disX, double disY, double disZ,
    double uTot, double vTot, double wTot,
    double txx, double tyy, double tzz,
    double txz, double txy, double tyz,
    double vs, double CoEps, double boxSizeZ, double nuT,
    double &m, double &m_kg,
    const double rho, const double d_m,
    const double c1, const double c2,
    const unsigned char depFlag,
    const int* icellflag,
    const double* z_faces,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double* dDepMass,
    bool &isActive
) {
  if (!isActive) return;
  if (!depFlag) return;
  if (!dDepMass) return;

  const double rhoAir = 1.225;
  const double nuAir  = 1.506e-5;
  const double pi     = 3.14159265358979323846;

  const double xPos_old = xPos - disX;
  const double yPos_old = yPos - disY;
  const double zPos_old = zPos - disZ;

  int cellId_old = gpu_getCellId(xPos_old, yPos_old, zPos_old, z_faces, nx, ny, nz, dx, dy, dz);
  int cellId     = gpu_getCellId(xPos,     yPos,     zPos,     z_faces, nx, ny, nz, dx, dy, dz);

  if (cellId_old < 0 || cellId < 0) return;

  int ii, jj, kk;
  gpu_getCellIndex_fromCellId(cellId, nx, ny, ii, jj, kk);

  const double partDist = sqrt(disX * disX + disY * disY + disZ * disZ);
  const double MTot     = sqrt(uTot * uTot + vTot * vTot + wTot * wTot);

  if (partDist <= 1.0e-20 || MTot <= 1.0e-20) return;

  // -------------------------
  // 1) canopy deposition
  // -------------------------
  if (gpu_isCanopyFlag(icellflag[cellId_old])) {
    const double elementDiameter  = 100.0e-3;
    const double leafAreaDensity  = 5.57;
    const double Cc               = 1.0;

    double tauSum = txx + tyy + tzz;
    if (tauSum < 0.0) tauSum = 0.0;

    const double parRMS = (1.0 / sqrt(3.0)) * sqrt(tauSum);
    const double CoEpsSafe = (CoEps > 1.0e-12) ? CoEps : 1.0e-12;
    const double taylorMicroscale =
        sqrt((15.0 * nuAir * 5.0 * parRMS * parRMS) / CoEpsSafe);

    const double Stk =
        (rho * d_m * d_m * MTot * Cc) /
        (18.0 * rhoAir * nuAir * elementDiameter);

    const double ReLambda = parRMS * taylorMicroscale / nuAir;

    double depEff = 0.0;
    double depArg = pow(pow(ReLambda, 0.3) * Stk, c2);
    depEff = 1.0 - 1.0 / (c1 * depArg + 1.0);

    double ReLeaf = elementDiameter * MTot / nuAir;
    if (ReLeaf > 6000.0) ReLeaf = 6000.0;
    if (ReLeaf < 400.0)  ReLeaf = 400.0;

    const double gam    = -6.5e-5 * ReLeaf + 0.43;
    const double adjLAD = leafAreaDensity * (1.0 + gam);
    const double P_v    = exp(-depEff * adjLAD * partDist * 0.7);

    const double depMass = (1.0 - P_v) * m;

    if (depMass > 0.0) {
      gpu_atomicAdd_double(&dDepMass[cellId_old], depMass);
    }

    m    *= P_v;
    m_kg *= P_v;
  }
  // -------------------------
  // 2) ground deposition
  // -------------------------
  else {
    const int plane   = (nx - 1) * (ny - 1);
    const int belowId = cellId - plane;

    if (belowId >= 0 && gpu_isTerrainFlag(icellflag[belowId])) {
      if (kk >= 1) {
        double dz_g = z_faces[kk] - z_faces[kk - 1];
        if (dz_g <= 1.0e-12) dz_g = boxSizeZ;

        const double dt_dep = partDist / MTot;

        double shear2 = txz * txz + tyz * tyz;
        if (shear2 < 1.0e-20) shear2 = 1.0e-20;

        const double ustarDep = pow(shear2, 0.25);

        double nuTSafe = (nuT > 1.0e-12) ? nuT : 1.0e-12;
        const double Sc = nuAir / nuTSafe;

        double ra = 1.0 / (0.4 * ustarDep) *
                    log(((10000.0 * ustarDep * boxSizeZ) / (2.0 * nuAir) + 1.0 / Sc) /
                        ((100.0   * ustarDep / nuAir) + 1.0 / Sc));

        double Stk_ground = (vs * ustarDep * ustarDep) / (9.81 * nuAir);
        if (Stk_ground < 1.0e-12) Stk_ground = 1.0e-12;

        double rb = 1.0 / (ustarDep *
                  (pow(Sc, -2.0 / 3.0) + pow(10.0, -3.0 / Stk_ground)));

        double denom = ra + rb + ra * rb * vs;
        if (denom < 1.0e-12) denom = 1.0e-12;

        double vd = 1.0 / denom + vs;
        double P_g = exp(-vd * dt_dep / dz_g);

        const double depMass = (1.0 - P_g) * m;

        if (depMass > 0.0) {
          gpu_atomicAdd_double(&dDepMass[cellId_old], depMass);
        }

        m    *= P_g;
        m_kg *= P_g;
      }
    }
  }

  // 和 CPU 一样：质量太小就失活
  const double oneParMass = rho * (1.0 / 6.0) * pi * d_m * d_m * d_m;
  if (m_kg < oneParMass) {
    m_kg = 0.0;
    m    = 0.0;
    isActive = false;
  }
}

__global__ void advect_langevin_kernel(
    int n,
    double dt,
    double* x,
    double* y,
    double* z,
    double* disX,
    double* disY,
    double* disZ,
    double* uMean,
    double* vMean,
    double* wMean,
    double* uFluct,
    double* vFluct,
    double* wFluct,
    double* uFluct_old,
    double* vFluct_old,
    double* wFluct_old,
    double* delta_uFluct,
    double* delta_vFluct,
    double* delta_wFluct,
    double* txx_old,
    double* txy_old,
    double* txz_old,
    double* tyy_old,
    double* tyz_old,
    double* tzz_old,
    double* CoEps,
    double* vs,
	double* m,
    double* m_kg,
    const double* rho,
    const double* d_m,
    const double* c1,
    const double* c2,
    const unsigned char* depFlag,
    unsigned char* isActive,
    unsigned char* isRogue,
    unsigned char* exitCode,
    int* stepActiveCount,
    int* stepInactiveCount,
    int* stepRogueCount,
    double boxSizeZ,
    int clampZ_and_deactivate,
    int use_const_wind,
    double const_u0,
    double const_v0,
    double const_w0,
    float gaussian_scale,
    double courantNum,
    double dxy,
    double dz,
    double sim_dt,
    double vel_threshold,
    const double* WGD_u,
    const double* WGD_v,
    const double* WGD_w,
    const int* WGD_icellflag,
    const double* WGD_x,
    const double* WGD_y,
    const double* TGD_CoEps,
    const double* TGD_txx,
    const double* TGD_txy,
    const double* TGD_txz,
    const double* TGD_tyy,
    const double* TGD_tyz,
    const double* TGD_tzz,
    const double* TGD_div_tau_x,
    const double* TGD_div_tau_y,
    const double* TGD_div_tau_z,
    const double* TGD_nuT,
    const double* WGD_mixingLengths,
    const double* z_nodes,
    const double* z_faces,
	double* dDepMass,
    int nx,
    int ny,
    int nz,
    double dx,
    double dy,
    double xStart,
    double xEnd,
    double yStart,
    double yEnd,
    double zStart,
    double zEnd,
    int bcTypeX,
    int bcTypeY,
    int bcTypeZ,
    double xBCStart,
    double xBCEnd,
    double yBCStart,
    double yBCEnd,
    double zBCStart,
    double zBCEnd,
    double domainZstart,
    curandStatePhilox4_32_10_t* rngStates
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  curandStatePhilox4_32_10_t localState = rngStates[i];

  bool active = (isActive[i] != 0);
  bool rogue  = (isRogue[i] != 0);
  unsigned char exitReason = GPU_EXIT_NONE;

  if (!active) {
    exitCode[i] = GPU_EXIT_NONE;
    rngStates[i] = localState;
    return;
  }


  double xPos = x[i];
  double yPos = y[i];
  double zPos = z[i];

  double disx = 0.0;
  double disy = 0.0;
  double disz = 0.0;

  double uMeanL = uMean[i];
  double vMeanL = vMean[i];
  double wMeanL = wMean[i];

  double uFluctL = uFluct[i];
  double vFluctL = vFluct[i];
  double wFluctL = wFluct[i];

  double uFluctOldL = uFluct_old[i];
  double vFluctOldL = vFluct_old[i];
   double wFluctOldL = wFluct_old[i];

  double delta_u = delta_uFluct[i];
  double delta_v = delta_vFluct[i];
  double delta_w = delta_wFluct[i];

  double txxOldL = txx_old[i];
  double txyOldL = txy_old[i];
  double txzOldL = txz_old[i];
  double tyyOldL = tyy_old[i];
  double tyzOldL = tyz_old[i];
  double tzzOldL = tzz_old[i];

  double CoEpsL = CoEps[i];
  if (CoEpsL <= 1.0e-12) CoEpsL = 1.0e-6;

  double mL    = m[i];
  double mKgL  = m_kg[i];

  const double rhoL = rho[i];
  const double d_mL = d_m[i];
  const double c1L  = c1[i];
  const double c2L  = c2[i];

  const unsigned char depFlagL = depFlag[i];

  double timeRemainder = dt;
  const double invarianceTol = 1.0e-12;

  int substep = 0;
  const int MAX_SUBSTEPS = 1024;

  while (active && timeRemainder > 1.0e-12 && substep < MAX_SUBSTEPS) {
    substep++;

    // ================================
    // 第一阶段严格对齐：
    // 不在子步一开始就做“起点是否在 BC 域内”的额外强杀。
    // CPU 原版是在位置更新 + 反射之后，再由 domainBC_x/y/z->enforce 处理边界。
    // 这里先不提前判死，避免和 CPU 逻辑时序不一致。
    // ================================
			
    const double uFluctPrev = uFluctOldL;
    const double vFluctPrev = vFluctOldL;
    const double wFluctPrev = wFluctOldL;

    const double txxPrev = txxOldL;
    const double txyPrev = txyOldL;
    const double txzPrev = txzOldL;
    const double tyyPrev = tyyOldL;
    const double tyzPrev = tyzOldL;
    const double tzzPrev = tzzOldL;

    double txx = txxOldL;
    double txy = txyOldL;
    double txz = txzOldL;
    double tyy = tyyOldL;
    double tyz = tyzOldL;
    double tzz = tzzOldL;

    double flux_div_x = 0.0;
    double flux_div_y = 0.0;
    double flux_div_z = 0.0;
    double nuT = 0.0;

    if (use_const_wind) {
      uMeanL = const_u0;
      vMeanL = const_v0;
      wMeanL = const_w0;
    } else {
      gpu_interpValuesTriLinear(
          xPos, yPos, zPos,
          WGD_u, WGD_v, WGD_w,
          TGD_CoEps,
          TGD_txx, TGD_txy, TGD_txz,
          TGD_tyy, TGD_tyz, TGD_tzz,
          TGD_div_tau_x, TGD_div_tau_y, TGD_div_tau_z,
          TGD_nuT,
          z_nodes, z_faces,
          nx, ny, nz, dx, dy,
          uMeanL, vMeanL, wMeanL,
          txx, txy, txz, tyy, tyz, tzz,
          flux_div_x, flux_div_y, flux_div_z,
          nuT, CoEpsL
      );
    }

    gpu_makeRealizable(txx, txy, txz, tyy, tyz, tzz, invarianceTol);

    // CPU 对齐：平均竖直速度直接减去粒子沉降速度
    double wMeanEff = wMeanL - vs[i];

    int cellId = gpu_getCellId(xPos, yPos, zPos, z_faces, nx, ny, nz, dx, dy, dz);
    double dWall = (double)WGD_mixingLengths[cellId];
    if (dWall <= 1.0e-12) dWall = fmin(dxy, dz);

    double par_dt = gpu_calcCourantTimestep_wall(
        dWall,
        fabs(uMeanL) + fabs(uFluctL),
        fabs(vMeanL) + fabs(vFluctL),
        fabs(wMeanEff) + fabs(wFluctL),
        timeRemainder,
        courantNum,
        dxy,
        dz,
        sim_dt
    );

    double lxx = txx;
    double lxy = txy;
    double lxz = txz;
    double lyx = txy;
    double lyy = tyy;
    double lyz = tyz;
    double lzx = txz;
    double lzy = tyz;
    double lzz = tzz;

    bool ok = gpu_invert3(lxx, lxy, lxz, lyx, lyy, lyz, lzx, lzy, lzz);
    if (!ok) {
      gpu_set_exit_reason(exitReason, GPU_EXIT_INV_SIGMA);
      rogue = true;
      active = false;
      break;
    }

    double dtxxdt = (txx - txxPrev) / ((par_dt > 1.0e-12) ? par_dt : 1.0e-12);
    double dtxydt = (txy - txyPrev) / ((par_dt > 1.0e-12) ? par_dt : 1.0e-12);
    double dtxzdt = (txz - txzPrev) / ((par_dt > 1.0e-12) ? par_dt : 1.0e-12);
    double dtyydt = (tyy - tyyPrev) / ((par_dt > 1.0e-12) ? par_dt : 1.0e-12);
    double dtyzdt = (tyz - tyzPrev) / ((par_dt > 1.0e-12) ? par_dt : 1.0e-12);
    double dtzzdt = (tzz - tzzPrev) / ((par_dt > 1.0e-12) ? par_dt : 1.0e-12);

    double xRandn = (double)gaussian_scale * curand_normal_double(&localState);
    double yRandn = (double)gaussian_scale * curand_normal_double(&localState);
    double zRandn = (double)gaussian_scale * curand_normal_double(&localState);

    double A_11 = -1.0 + 0.50 * (-CoEpsL * lxx + lxx * dtxxdt + lxy * dtxydt + lxz * dtxzdt) * par_dt;
    double A_12 =  0.50 * (-CoEpsL * lxy + lxy * dtxxdt + lyy * dtxydt + lyz * dtxzdt) * par_dt;
    double A_13 =  0.50 * (-CoEpsL * lxz + lxz * dtxxdt + lyz * dtxydt + lzz * dtxzdt) * par_dt;

    double A_21 =  0.50 * (-CoEpsL * lxy + lxx * dtxydt + lxy * dtyydt + lxz * dtyzdt) * par_dt;
    double A_22 = -1.0 + 0.50 * (-CoEpsL * lyy + lxy * dtxydt + lyy * dtyydt + lyz * dtyzdt) * par_dt;
    double A_23 =  0.50 * (-CoEpsL * lyz + lxz * dtxydt + lyz * dtyydt + lzz * dtyzdt) * par_dt;

    double A_31 =  0.50 * (-CoEpsL * lxz + lxx * dtxzdt + lxy * dtyzdt + lxz * dtzzdt) * par_dt;
    double A_32 =  0.50 * (-CoEpsL * lyz + lxy * dtxzdt + lyy * dtyzdt + lyz * dtzzdt) * par_dt;
    double A_33 = -1.0 + 0.50 * (-CoEpsL * lzz + lxz * dtxzdt + lyz * dtyzdt + lzz * dtzzdt) * par_dt;

    const double randCoeff = sqrt(CoEpsL * par_dt);

    double b_11 = -uFluctPrev - 0.50 * flux_div_x * par_dt - randCoeff * xRandn;
    double b_21 = -vFluctPrev - 0.50 * flux_div_y * par_dt - randCoeff * yRandn;
    double b_31 = -wFluctPrev - 0.50 * flux_div_z * par_dt - randCoeff * zRandn;

    ok = gpu_invert3(A_11, A_12, A_13, A_21, A_22, A_23, A_31, A_32, A_33);
    if (!ok) {
      gpu_set_exit_reason(exitReason, GPU_EXIT_INV_A);
      rogue = true;
      active = false;
      break;
    }

    gpu_matmult(
        A_11, A_12, A_13,
        A_21, A_22, A_23,
        A_31, A_32, A_33,
        b_11, b_21, b_31,
        uFluctL, vFluctL, wFluctL
    );

    if (fabs(uFluctL) >= vel_threshold || isnan(uFluctL)) {
      gpu_set_exit_reason(exitReason, GPU_EXIT_ROGUE_U);
      uFluctL = 0.0;
      rogue = true;
      active = false;
      break;
    }
    if (fabs(vFluctL) >= vel_threshold || isnan(vFluctL)) {
      gpu_set_exit_reason(exitReason, GPU_EXIT_ROGUE_V);
      vFluctL = 0.0;
      rogue = true;
      active = false;
      break;
    }
    if (fabs(wFluctL) >= vel_threshold || isnan(wFluctL)) {
      gpu_set_exit_reason(exitReason, GPU_EXIT_ROGUE_W);
      wFluctL = 0.0;
      rogue = true;
      active = false;
      break;
    }

    disx = (uMeanL + uFluctL) * par_dt;
    disy = (vMeanL + vFluctL) * par_dt;
    disz = (wMeanEff + wFluctL) * par_dt;

    xPos += disx;
    yPos += disy;
    zPos += disz;
	
    double uTot = uMeanL + uFluctL;
    double vTot = vMeanL + vFluctL;
    double wTot = wMeanEff + wFluctL;

    // CPU 顺序对齐：先沉积，再 wall reflection，再 domain BC
    if (active && !use_const_wind) {
      gpu_apply_deposition(
          xPos, yPos, zPos,
          disx, disy, disz,
          uTot, vTot, wTot,
          txx, tyy, tzz,
          txz, txy, tyz,
          vs[i], CoEpsL, boxSizeZ, nuT,
          mL, mKgL,
          rhoL, d_mL, c1L, c2L,
          depFlagL,
          WGD_icellflag,
          z_faces,
          nx, ny, nz,
          dx, dy, dz,
          dDepMass,
          active
      );
    }
	
// ------------------------------------------------------------
// CPU 对齐顺序：
// 1) 先 wall reflection
// 2) 再 domain BC
// 3) 最后只对“仍在物理域内”的粒子做 final solid-cell 检查
// ------------------------------------------------------------

// 1) wall reflection
if (active && !use_const_wind) {
  bool okReflect = gpu_reflect_stairstep_full(
      WGD_icellflag,
      WGD_x,
      WGD_y,
      z_faces,
      nx, ny, nz,
      dx, dy, dz,
      domainZstart,
      xPos, yPos, zPos,
      disx, disy, disz,
      uFluctL, vFluctL, wFluctL,
      exitReason
  );
  if (!okReflect) {
    if (exitReason == GPU_EXIT_NONE) {
      gpu_set_exit_reason(exitReason, GPU_EXIT_REFLECT_FAIL);
    }
    active = false;
  }
}

// 2) domain BC
// 完全对齐 CPU：调用与 DomainBC_*::enforce 同语义的 GPU 版本
if (active) {
  bool okBCx = gpu_enforce_domainBC_1d(
      xPos, uFluctL,
      xStart, xEnd,
      bcTypeX
  );
  if (!okBCx) {
    gpu_set_exit_reason(exitReason, GPU_EXIT_X_BC);
    active = false;
  }
}

if (active) {
  bool okBCy = gpu_enforce_domainBC_1d(
      yPos, vFluctL,
      yStart, yEnd,
      bcTypeY
  );
  if (!okBCy) {
    gpu_set_exit_reason(exitReason, GPU_EXIT_Y_BC);
    active = false;
  }
}

if (active) {
  bool okBCz = gpu_enforce_domainBC_1d(
      zPos, wFluctL,
      zStart, zEnd,
      bcTypeZ
  );
  if (!okBCz) {
    gpu_set_exit_reason(exitReason, GPU_EXIT_Z_BC);
    active = false;
  }
}

// 3) final solid / terrain check
// 只对“边界条件之后仍存活、且仍在物理域内”的粒子做这个检查，
// 避免把真正的 boundary exit 误记成 end-invalid/solid-cell。
if (active && !use_const_wind) {
  int finalCellId, finalCellFlag;
  GpuVec3d Xfinal(xPos, yPos, zPos);
  bool okFinalCell = gpu_cellflag_at(
      WGD_icellflag, z_faces,
      nx, ny, nz, dx, dy, dz, domainZstart,
      Xfinal, finalCellId, finalCellFlag);

  if (!okFinalCell || finalCellFlag == 0 || finalCellFlag == 2) {
    gpu_set_exit_reason(exitReason, GPU_EXIT_END_INVALID_CELL);
    active = false;
  }
}
    if (!active) {
      break;
    }

    delta_u = uFluctL - uFluctPrev;
    delta_v = vFluctL - vFluctPrev;
    delta_w = wFluctL - wFluctPrev;

    uFluctOldL = uFluctL;
    vFluctOldL = vFluctL;
    wFluctOldL = wFluctL;

    txxOldL = txx;
    txyOldL = txy;
    txzOldL = txz;
    tyyOldL = tyy;
    tyzOldL = tyz;
    tzzOldL = tzz;

    timeRemainder -= par_dt;
  }

  x[i] = xPos;
  y[i] = yPos;
  z[i] = zPos;

  disX[i] = disx;
  disY[i] = disy;
  disZ[i] = disz;

  uMean[i] = uMeanL;
  vMean[i] = vMeanL;
  wMean[i] = wMeanL;

  uFluct[i] = uFluctL;
  vFluct[i] = vFluctL;
  wFluct[i] = wFluctL;

  uFluct_old[i] = uFluctOldL;
  vFluct_old[i] = vFluctOldL;
  wFluct_old[i] = wFluctOldL;

  delta_uFluct[i] = delta_u;
  delta_vFluct[i] = delta_v;
  delta_wFluct[i] = delta_w;

  txx_old[i] = txxOldL;
  txy_old[i] = txyOldL;
  txz_old[i] = txzOldL;
  tyy_old[i] = tyyOldL;
  tyz_old[i] = tyzOldL;
  tzz_old[i] = tzzOldL;

  m[i]    = mL;
  m_kg[i] = mKgL;

  CoEps[i] = CoEpsL;
  isActive[i] = active ? 1 : 0;
  isRogue[i]  = rogue  ? 1 : 0;
  exitCode[i] = exitReason;

  if (active) {
    atomicAdd(stepActiveCount, 1);
  } else {
    atomicAdd(stepInactiveCount, 1);
  }

  if (rogue) {
    atomicAdd(stepRogueCount, 1);
  }

  rngStates[i] = localState;
}

__global__ void accumulate_eulerian_boxes_kernel(
    int n,
    const double* x,
    const double* y,
    const double* z,
    const double* m,
    const double* wdecay,
    const unsigned char* isActive,
    int nBoxesX,
    int nBoxesY,
    int nBoxesZ,
    float lBndx,
    float lBndy,
    float lBndz,
    float boxSizeX,
    float boxSizeY,
    float boxSizeZ,
    float dt,
    int* dPBox,
    float* dConc)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (!isActive[i]) return;

  const double xi = x[i];
  const double yi = y[i];
  const double zi = z[i];

  if (!isfinite(xi) || !isfinite(yi) || !isfinite(zi)) return;

  int idx = (int)floor((xi - lBndx) / (boxSizeX + 1.0e-9f));
  int idy = (int)floor((yi - lBndy) / (boxSizeY + 1.0e-9f));
  int idz = (int)floor((zi - lBndz) / (boxSizeZ + 1.0e-9f));

  if (idx < 0 || idx >= nBoxesX) return;
  if (idy < 0 || idy >= nBoxesY) return;
  if (idz < 0 || idz >= nBoxesZ) return;

  const double wi = (wdecay != nullptr) ? wdecay[i] : 1.0;
  const double contrib = m[i] * wi * (double)dt;

  if (!isfinite(contrib) || contrib <= 0.0) return;

  const int cellId = idx + idy * nBoxesX + idz * nBoxesX * nBoxesY;

  atomicAdd(&dPBox[cellId], 1);
  atomicAdd(&dConc[cellId], (float)contrib);
}

struct GpuGridCache {
  bool ready = false;
  int nx = 0, ny = 0, nz = 0;
  size_t sz_u = 0, sz_v = 0, sz_w = 0;
  size_t sz_cell = 0, sz_turb = 0, sz_z = 0;
  size_t sz_dep = 0;

  double *dWGD_u = nullptr, *dWGD_v = nullptr, *dWGD_w = nullptr;
  int *dIcellflag = nullptr;
  double *dXcenters = nullptr, *dYcenters = nullptr;
  double *dTGD_CoEps = nullptr;
  double *dTGD_txx = nullptr, *dTGD_txy = nullptr, *dTGD_txz = nullptr;
  double *dTGD_tyy = nullptr, *dTGD_tyz = nullptr, *dTGD_tzz = nullptr;
  double *dTGD_div_tau_x = nullptr, *dTGD_div_tau_y = nullptr, *dTGD_div_tau_z = nullptr;
  double *dTGD_nuT = nullptr;
  double *dMixingLengths = nullptr;
  double *dz_nodes = nullptr, *dz_faces = nullptr;

  double *dDepMass = nullptr;
};

struct GpuParticleCache {
  int cap = 0;
  int n_used = 0;
  bool uploaded = false;
  bool hostDirty = false;

  double *dx_ptr = nullptr, *dy_ptr = nullptr, *dz_ptr = nullptr;
  double *ddisX = nullptr, *ddisY = nullptr, *ddisZ = nullptr;
  double *duMean = nullptr, *dvMean = nullptr, *dwMean = nullptr;
  double *duFluct = nullptr, *dvFluct = nullptr, *dwFluct = nullptr;
  double *duFluct_old = nullptr, *dvFluct_old = nullptr, *dwFluct_old = nullptr;
  double *ddelta_uFluct = nullptr, *ddelta_vFluct = nullptr, *ddelta_wFluct = nullptr;
  double *dtxx_old = nullptr, *dtxy_old = nullptr, *dtxz_old = nullptr;
  double *dtyy_old = nullptr, *dtyz_old = nullptr, *dtzz_old = nullptr;
  double *dCoEps = nullptr, *dvs = nullptr;

  double *dm = nullptr, *dm_kg = nullptr;
  double *dwdecay = nullptr;
  double *drho = nullptr, *dd_m = nullptr;
  double *dc1 = nullptr, *dc2 = nullptr;
  unsigned char *dDepFlag = nullptr;

  unsigned char *dActive = nullptr, *dRogue = nullptr, *dExitCode = nullptr;
  curandStatePhilox4_32_10_t* dStates = nullptr;
};

static GpuGridCache gGrid;
static GpuParticleCache gPart;

struct GpuEulerianBoxCache {
  bool ready = false;

  int nBoxesX = 0;
  int nBoxesY = 0;
  int nBoxesZ = 0;
  int nCells  = 0;

  float lBndx = 0.0f;
  float lBndy = 0.0f;
  float lBndz = 0.0f;

  float boxSizeX = 0.0f;
  float boxSizeY = 0.0f;
  float boxSizeZ = 0.0f;

  int*   dPBox = nullptr;
  float* dConc = nullptr;   // 这里直接累计 m * dt，最后仍然在 host 端除以 (Tavg * volume)
};

static GpuEulerianBoxCache gBox;

struct GpuStepStats {
  int *dActiveCount   = nullptr;
  int *dInactiveCount = nullptr;
  int *dRogueCount    = nullptr;

  int hActiveCount    = 0;
  int hInactiveCount  = 0;
  int hRogueCount     = 0;
};

static GpuStepStats gStep;
static std::vector<double> gHostDepBuffer;

template <typename T>
static void freeDeviceArray(T*& p) {
  if (p) {
    cudaFree(p);
    p = nullptr;
  }
}

static void ensureStepStats() {
  if (!gStep.dActiveCount) {
    cudaCheck(cudaMalloc(&gStep.dActiveCount, sizeof(int)), "malloc step active");
  }
  if (!gStep.dInactiveCount) {
    cudaCheck(cudaMalloc(&gStep.dInactiveCount, sizeof(int)), "malloc step inactive");
  }
  if (!gStep.dRogueCount) {
    cudaCheck(cudaMalloc(&gStep.dRogueCount, sizeof(int)), "malloc step rogue");
  }
}

static void freeStepStats() {
  freeDeviceArray(gStep.dActiveCount);
  freeDeviceArray(gStep.dInactiveCount);
  freeDeviceArray(gStep.dRogueCount);
  gStep = GpuStepStats();
}

void advectParticlesGPU_demo_getLastStepStats(int &activeCount,
                                              int &inactiveCount,
                                              int &rogueCount) {
  activeCount   = gStep.hActiveCount;
  inactiveCount = gStep.hInactiveCount;
  rogueCount    = gStep.hRogueCount;
}

static void gpu_print_exit_window_summary(
    const long long* counts,
    int callCount,
    double dtSum,
    int activeNow
) {
  std::cout << "[GPU-DIAG] =============================================\n";
  std::cout << "[GPU-DIAG] 最近一个时间窗口: " << callCount
            << " 个时间步, 累计 dt = " << dtSum << " s\n";
  long long total = 0;
  for (int code = 1; code <= GPU_EXIT_CODE_MAX; ++code) total += counts[code];
  std::cout << "[GPU-DIAG] 本窗口新失活粒子数 = " << total << "\n";
  std::cout << "[GPU-DIAG] 本窗口失活原因统计:\n";
  for (int code = 1; code <= GPU_EXIT_CODE_MAX; ++code) {
    if (counts[code] > 0) {
      std::cout << "  code=" << code << "  "
                << gpu_exit_reason_name((unsigned char)code)
                << " : " << counts[code] << "\n";
    }
  }
  std::cout << "[GPU-DIAG] 当前 active = " << activeNow << "\n";
  std::cout << "[GPU-DIAG] =============================================\n";
}

static void freeEulerianBoxCache() {
  freeDeviceArray(gBox.dPBox);
  freeDeviceArray(gBox.dConc);
  gBox = GpuEulerianBoxCache();
}

template <typename T>
static void growDeviceArrayPreserve(T*& ptr, int oldCap, int newCap, const char* what) {
  T* newPtr = nullptr;
  cudaCheck(cudaMalloc(&newPtr, newCap * sizeof(T)), what);

  if (ptr && oldCap > 0) {
    cudaCheck(
        cudaMemcpy(newPtr, ptr, oldCap * sizeof(T), cudaMemcpyDeviceToDevice),
        "growDeviceArrayPreserve D2D");
    cudaFree(ptr);
  }

  ptr = newPtr;
}

static void freeParticleCache() {
  freeDeviceArray(gPart.dx_ptr);
  freeDeviceArray(gPart.dy_ptr);
  freeDeviceArray(gPart.dz_ptr);

  freeDeviceArray(gPart.ddisX);
  freeDeviceArray(gPart.ddisY);
  freeDeviceArray(gPart.ddisZ);

  freeDeviceArray(gPart.duMean);
  freeDeviceArray(gPart.dvMean);
  freeDeviceArray(gPart.dwMean);

  freeDeviceArray(gPart.duFluct);
  freeDeviceArray(gPart.dvFluct);
  freeDeviceArray(gPart.dwFluct);

  freeDeviceArray(gPart.duFluct_old);
  freeDeviceArray(gPart.dvFluct_old);
  freeDeviceArray(gPart.dwFluct_old);

  freeDeviceArray(gPart.ddelta_uFluct);
  freeDeviceArray(gPart.ddelta_vFluct);
  freeDeviceArray(gPart.ddelta_wFluct);

  freeDeviceArray(gPart.dtxx_old);
  freeDeviceArray(gPart.dtxy_old);
  freeDeviceArray(gPart.dtxz_old);
  freeDeviceArray(gPart.dtyy_old);
  freeDeviceArray(gPart.dtyz_old);
  freeDeviceArray(gPart.dtzz_old);

  freeDeviceArray(gPart.dCoEps);
  freeDeviceArray(gPart.dvs);
  
  freeDeviceArray(gPart.dm);
  freeDeviceArray(gPart.dm_kg);
  freeDeviceArray(gPart.drho);
  freeDeviceArray(gPart.dd_m);
  freeDeviceArray(gPart.dc1);
  freeDeviceArray(gPart.dc2);
  freeDeviceArray(gPart.dDepFlag);

  freeDeviceArray(gPart.dActive);
  freeDeviceArray(gPart.dRogue);
  freeDeviceArray(gPart.dExitCode);

  freeDeviceArray(gPart.dStates);

  gPart = GpuParticleCache();
}

static void freeGridCache() {
  freeDeviceArray(gGrid.dWGD_u);
  freeDeviceArray(gGrid.dWGD_v);
  freeDeviceArray(gGrid.dWGD_w);

  freeDeviceArray(gGrid.dIcellflag);
  freeDeviceArray(gGrid.dXcenters);
  freeDeviceArray(gGrid.dYcenters);

  freeDeviceArray(gGrid.dTGD_CoEps);

  freeDeviceArray(gGrid.dTGD_txx);
  freeDeviceArray(gGrid.dTGD_txy);
  freeDeviceArray(gGrid.dTGD_txz);
  freeDeviceArray(gGrid.dTGD_tyy);
  freeDeviceArray(gGrid.dTGD_tyz);
  freeDeviceArray(gGrid.dTGD_tzz);

  freeDeviceArray(gGrid.dTGD_div_tau_x);
  freeDeviceArray(gGrid.dTGD_div_tau_y);
  freeDeviceArray(gGrid.dTGD_div_tau_z);

  freeDeviceArray(gGrid.dTGD_nuT);
  freeDeviceArray(gGrid.dMixingLengths);
  freeDeviceArray(gGrid.dz_nodes);
  freeDeviceArray(gGrid.dz_faces);
  
  freeDeviceArray(gGrid.dDepMass);

  gGrid = GpuGridCache();
}

void advectParticlesGPU_demo_resetCache() {
  freeParticleCache();
  freeGridCache();
  freeEulerianBoxCache();
  gHostDepBuffer.clear();
  gHostDepBuffer.shrink_to_fit();
}

static void ensureParticleCache(int n) {
  if (gPart.cap >= n) return;

  const int oldCap = gPart.cap;
  int newCap = (oldCap > 0) ? oldCap : 1;

  while (newCap < n) {
    newCap = newCap + newCap / 2 + 1024;
  }
  if (newCap < n) newCap = n;

  growDeviceArrayPreserve(gPart.dx_ptr, oldCap, newCap, "grow x");
  growDeviceArrayPreserve(gPart.dy_ptr, oldCap, newCap, "grow y");
  growDeviceArrayPreserve(gPart.dz_ptr, oldCap, newCap, "grow z");

  growDeviceArrayPreserve(gPart.ddisX, oldCap, newCap, "grow disX");
  growDeviceArrayPreserve(gPart.ddisY, oldCap, newCap, "grow disY");
  growDeviceArrayPreserve(gPart.ddisZ, oldCap, newCap, "grow disZ");

  growDeviceArrayPreserve(gPart.duMean, oldCap, newCap, "grow uMean");
  growDeviceArrayPreserve(gPart.dvMean, oldCap, newCap, "grow vMean");
  growDeviceArrayPreserve(gPart.dwMean, oldCap, newCap, "grow wMean");

  growDeviceArrayPreserve(gPart.duFluct, oldCap, newCap, "grow uFluct");
  growDeviceArrayPreserve(gPart.dvFluct, oldCap, newCap, "grow vFluct");
  growDeviceArrayPreserve(gPart.dwFluct, oldCap, newCap, "grow wFluct");

  growDeviceArrayPreserve(gPart.duFluct_old, oldCap, newCap, "grow uFluct_old");
  growDeviceArrayPreserve(gPart.dvFluct_old, oldCap, newCap, "grow vFluct_old");
  growDeviceArrayPreserve(gPart.dwFluct_old, oldCap, newCap, "grow wFluct_old");

  growDeviceArrayPreserve(gPart.ddelta_uFluct, oldCap, newCap, "grow delta_uFluct");
  growDeviceArrayPreserve(gPart.ddelta_vFluct, oldCap, newCap, "grow delta_vFluct");
  growDeviceArrayPreserve(gPart.ddelta_wFluct, oldCap, newCap, "grow delta_wFluct");

  growDeviceArrayPreserve(gPart.dtxx_old, oldCap, newCap, "grow txx_old");
  growDeviceArrayPreserve(gPart.dtxy_old, oldCap, newCap, "grow txy_old");
  growDeviceArrayPreserve(gPart.dtxz_old, oldCap, newCap, "grow txz_old");
  growDeviceArrayPreserve(gPart.dtyy_old, oldCap, newCap, "grow tyy_old");
  growDeviceArrayPreserve(gPart.dtyz_old, oldCap, newCap, "grow tyz_old");
  growDeviceArrayPreserve(gPart.dtzz_old, oldCap, newCap, "grow tzz_old");

  growDeviceArrayPreserve(gPart.dCoEps, oldCap, newCap, "grow CoEps");
  growDeviceArrayPreserve(gPart.dvs, oldCap, newCap, "grow vs");
  
  growDeviceArrayPreserve(gPart.dm,      oldCap, newCap, "grow m");
  growDeviceArrayPreserve(gPart.dm_kg,   oldCap, newCap, "grow m_kg");
  growDeviceArrayPreserve(gPart.dwdecay, oldCap, newCap, "grow wdecay");
  growDeviceArrayPreserve(gPart.drho,   oldCap, newCap, "grow rho");
  growDeviceArrayPreserve(gPart.dd_m,   oldCap, newCap, "grow d_m");
  growDeviceArrayPreserve(gPart.dc1,    oldCap, newCap, "grow c1");
  growDeviceArrayPreserve(gPart.dc2,    oldCap, newCap, "grow c2");
  growDeviceArrayPreserve(gPart.dDepFlag, oldCap, newCap, "grow depFlag");

  growDeviceArrayPreserve(gPart.dActive, oldCap, newCap, "grow isActive");
  growDeviceArrayPreserve(gPart.dRogue, oldCap, newCap, "grow isRogue");
  growDeviceArrayPreserve(gPart.dExitCode, oldCap, newCap, "grow exitCode");

  growDeviceArrayPreserve(gPart.dStates, oldCap, newCap, "grow rngStates");

  if (newCap > oldCap) {
    int threads = 256;
    int countNew = newCap - oldCap;
    int blocks = (countNew + threads - 1) / threads;

    init_rng_states_range_kernel<<<blocks, threads>>>(gPart.dStates, oldCap, newCap, 1234ULL);
    cudaCheck(cudaGetLastError(), "init rng range kernel");
	cudaCheck(cudaDeviceSynchronize(), "init rng range sync");
  }

  gPart.cap = newCap;
}

static void ensureGridCache(WINDSGeneralData* WGD, TURBGeneralData* TGD) {
  int nx = WGD->nx, ny = WGD->ny, nz = WGD->nz;
  size_t sz_u = WGD->u.size(), sz_v = WGD->v.size(), sz_w = WGD->w.size();
  size_t sz_cell = WGD->icellflag.size();
  size_t sz_turb = TGD->CoEps.size();
  size_t sz_z = WGD->z.size();
  size_t sz_dep = sz_cell;

  bool need = (!gGrid.ready || gGrid.nx != nx || gGrid.ny != ny || gGrid.nz != nz ||
               gGrid.sz_u != sz_u || gGrid.sz_v != sz_v || gGrid.sz_w != sz_w ||
               gGrid.sz_cell != sz_cell || gGrid.sz_turb != sz_turb || gGrid.sz_z != sz_z);
  if (!need) return;

  if (gGrid.dWGD_u) {
    cudaFree(gGrid.dWGD_u); cudaFree(gGrid.dWGD_v); cudaFree(gGrid.dWGD_w);
    cudaFree(gGrid.dIcellflag); cudaFree(gGrid.dXcenters); cudaFree(gGrid.dYcenters);
    cudaFree(gGrid.dTGD_CoEps);
    cudaFree(gGrid.dTGD_txx); cudaFree(gGrid.dTGD_txy); cudaFree(gGrid.dTGD_txz);
    cudaFree(gGrid.dTGD_tyy); cudaFree(gGrid.dTGD_tyz); cudaFree(gGrid.dTGD_tzz);
    cudaFree(gGrid.dTGD_div_tau_x); cudaFree(gGrid.dTGD_div_tau_y); cudaFree(gGrid.dTGD_div_tau_z);
    cudaFree(gGrid.dTGD_nuT); cudaFree(gGrid.dMixingLengths); cudaFree(gGrid.dz_nodes); cudaFree(gGrid.dz_faces);
    gGrid = GpuGridCache();
  }

  std::vector<double> hWGD_u(WGD->u.begin(), WGD->u.end());
  std::vector<double> hWGD_v(WGD->v.begin(), WGD->v.end());
  std::vector<double> hWGD_w(WGD->w.begin(), WGD->w.end());
  std::vector<double> hMixingLengths(WGD->mixingLengths.begin(), WGD->mixingLengths.end());
  std::vector<int> hIcellflag(WGD->icellflag.begin(), WGD->icellflag.end());
  std::vector<double> hXcenters(WGD->x.begin(), WGD->x.end());
  std::vector<double> hYcenters(WGD->y.begin(), WGD->y.end());
  std::vector<double> hTGD_CoEps(TGD->CoEps.begin(), TGD->CoEps.end());
  std::vector<double> hTGD_txx(TGD->txx.begin(), TGD->txx.end());
  std::vector<double> hTGD_txy(TGD->txy.begin(), TGD->txy.end());
  std::vector<double> hTGD_txz(TGD->txz.begin(), TGD->txz.end());
  std::vector<double> hTGD_tyy(TGD->tyy.begin(), TGD->tyy.end());
  std::vector<double> hTGD_tyz(TGD->tyz.begin(), TGD->tyz.end());
  std::vector<double> hTGD_tzz(TGD->tzz.begin(), TGD->tzz.end());
  std::vector<double> hTGD_div_tau_x(TGD->div_tau_x.begin(), TGD->div_tau_x.end());
  std::vector<double> hTGD_div_tau_y(TGD->div_tau_y.begin(), TGD->div_tau_y.end());
  std::vector<double> hTGD_div_tau_z(TGD->div_tau_z.begin(), TGD->div_tau_z.end());
  std::vector<double> hTGD_nuT(TGD->nuT.begin(), TGD->nuT.end());
  std::vector<double> hz_nodes(WGD->z.begin(), WGD->z.end());
  std::vector<double> hz_faces(WGD->z_face.begin(), WGD->z_face.end());

  cudaCheck(cudaMalloc(&gGrid.dWGD_u, hWGD_u.size() * sizeof(double)), "malloc WGD_u");
  cudaCheck(cudaMalloc(&gGrid.dWGD_v, hWGD_v.size() * sizeof(double)), "malloc WGD_v");
  cudaCheck(cudaMalloc(&gGrid.dWGD_w, hWGD_w.size() * sizeof(double)), "malloc WGD_w");
  cudaCheck(cudaMalloc(&gGrid.dIcellflag, hIcellflag.size() * sizeof(int)), "malloc icellflag");
  cudaCheck(cudaMalloc(&gGrid.dXcenters, hXcenters.size() * sizeof(double)), "malloc xCenters");
  cudaCheck(cudaMalloc(&gGrid.dYcenters, hYcenters.size() * sizeof(double)), "malloc yCenters");
  cudaCheck(cudaMalloc(&gGrid.dTGD_CoEps, hTGD_CoEps.size() * sizeof(double)), "malloc TGD_CoEps");
  cudaCheck(cudaMalloc(&gGrid.dTGD_txx, hTGD_txx.size() * sizeof(double)), "malloc TGD_txx");
  cudaCheck(cudaMalloc(&gGrid.dTGD_txy, hTGD_txy.size() * sizeof(double)), "malloc TGD_txy");
  cudaCheck(cudaMalloc(&gGrid.dTGD_txz, hTGD_txz.size() * sizeof(double)), "malloc TGD_txz");
  cudaCheck(cudaMalloc(&gGrid.dTGD_tyy, hTGD_tyy.size() * sizeof(double)), "malloc TGD_tyy");
  cudaCheck(cudaMalloc(&gGrid.dTGD_tyz, hTGD_tyz.size() * sizeof(double)), "malloc TGD_tyz");
  cudaCheck(cudaMalloc(&gGrid.dTGD_tzz, hTGD_tzz.size() * sizeof(double)), "malloc TGD_tzz");
  cudaCheck(cudaMalloc(&gGrid.dTGD_div_tau_x, hTGD_div_tau_x.size() * sizeof(double)), "malloc div_tau_x");
  cudaCheck(cudaMalloc(&gGrid.dTGD_div_tau_y, hTGD_div_tau_y.size() * sizeof(double)), "malloc div_tau_y");
  cudaCheck(cudaMalloc(&gGrid.dTGD_div_tau_z, hTGD_div_tau_z.size() * sizeof(double)), "malloc div_tau_z");
  cudaCheck(cudaMalloc(&gGrid.dTGD_nuT, hTGD_nuT.size() * sizeof(double)), "malloc nuT");
  cudaCheck(cudaMalloc(&gGrid.dMixingLengths, hMixingLengths.size() * sizeof(double)), "malloc mixingLengths");
  cudaCheck(cudaMalloc(&gGrid.dz_nodes, hz_nodes.size() * sizeof(double)), "malloc z_nodes");
  cudaCheck(cudaMalloc(&gGrid.dz_faces, hz_faces.size() * sizeof(double)), "malloc z_faces");
  cudaCheck(cudaMalloc(&gGrid.dDepMass, sz_dep * sizeof(double)), "malloc dDepMass");
  cudaCheck(cudaMemset(gGrid.dDepMass, 0, sz_dep * sizeof(double)), "memset dDepMass");

  cudaCheck(cudaMemcpy(gGrid.dWGD_u, hWGD_u.data(), hWGD_u.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D WGD_u");
  cudaCheck(cudaMemcpy(gGrid.dWGD_v, hWGD_v.data(), hWGD_v.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D WGD_v");
  cudaCheck(cudaMemcpy(gGrid.dWGD_w, hWGD_w.data(), hWGD_w.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D WGD_w");
  cudaCheck(cudaMemcpy(gGrid.dIcellflag, hIcellflag.data(), hIcellflag.size() * sizeof(int), cudaMemcpyHostToDevice), "H2D icellflag");
  cudaCheck(cudaMemcpy(gGrid.dXcenters, hXcenters.data(), hXcenters.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D xCenters");
  cudaCheck(cudaMemcpy(gGrid.dYcenters, hYcenters.data(), hYcenters.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D yCenters");
  cudaCheck(cudaMemcpy(gGrid.dTGD_CoEps, hTGD_CoEps.data(), hTGD_CoEps.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D TGD_CoEps");
  cudaCheck(cudaMemcpy(gGrid.dTGD_txx, hTGD_txx.data(), hTGD_txx.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D TGD_txx");
  cudaCheck(cudaMemcpy(gGrid.dTGD_txy, hTGD_txy.data(), hTGD_txy.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D TGD_txy");
  cudaCheck(cudaMemcpy(gGrid.dTGD_txz, hTGD_txz.data(), hTGD_txz.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D TGD_txz");
  cudaCheck(cudaMemcpy(gGrid.dTGD_tyy, hTGD_tyy.data(), hTGD_tyy.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D TGD_tyy");
  cudaCheck(cudaMemcpy(gGrid.dTGD_tyz, hTGD_tyz.data(), hTGD_tyz.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D TGD_tyz");
  cudaCheck(cudaMemcpy(gGrid.dTGD_tzz, hTGD_tzz.data(), hTGD_tzz.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D TGD_tzz");
  cudaCheck(cudaMemcpy(gGrid.dTGD_div_tau_x, hTGD_div_tau_x.data(), hTGD_div_tau_x.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D div_tau_x");
  cudaCheck(cudaMemcpy(gGrid.dTGD_div_tau_y, hTGD_div_tau_y.data(), hTGD_div_tau_y.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D div_tau_y");
  cudaCheck(cudaMemcpy(gGrid.dTGD_div_tau_z, hTGD_div_tau_z.data(), hTGD_div_tau_z.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D div_tau_z");
  cudaCheck(cudaMemcpy(gGrid.dTGD_nuT, hTGD_nuT.data(), hTGD_nuT.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D nuT");
  cudaCheck(cudaMemcpy(gGrid.dMixingLengths, hMixingLengths.data(), hMixingLengths.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D mixingLengths");
  cudaCheck(cudaMemcpy(gGrid.dz_nodes, hz_nodes.data(), hz_nodes.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D z_nodes");
  cudaCheck(cudaMemcpy(gGrid.dz_faces, hz_faces.data(), hz_faces.size() * sizeof(double), cudaMemcpyHostToDevice), "H2D z_faces");

  gGrid.ready = true; gGrid.nx = nx; gGrid.ny = ny; gGrid.nz = nz;
  gGrid.sz_u = sz_u; gGrid.sz_v = sz_v; gGrid.sz_w = sz_w; gGrid.sz_cell = sz_cell; gGrid.sz_turb = sz_turb; gGrid.sz_z = sz_z;
  gGrid.sz_dep = sz_dep;
}


struct GpuDiagWindow {
  long long exitCounts[32];
  long long deactivatedZBins[4];
  long long activeZBins[4];
  int callCount = 0;
  double windowDt = 0.0;
  long long newlyDeactivated = 0;
  long long activeBeforeLast = 0;
  long long activeAfterLast = 0;
  double maxActiveZLast = 0.0;
  double meanActiveZLast = 0.0;
  std::vector<std::string> sampleLines;

  GpuDiagWindow() { reset(); }

  void reset() {
    for (int i = 0; i < 32; ++i) exitCounts[i] = 0;
    for (int i = 0; i < 4; ++i) {
      deactivatedZBins[i] = 0;
      activeZBins[i] = 0;
    }
    callCount = 0;
    windowDt = 0.0;
    newlyDeactivated = 0;
    activeBeforeLast = 0;
    activeAfterLast = 0;
    maxActiveZLast = 0.0;
    meanActiveZLast = 0.0;
    sampleLines.clear();
  }
};

struct GpuDiagOverall {
  long long totalExitCounts[32];
  long long totalWindows = 0;
  long long totalCalls = 0;
  double totalDt = 0.0;
  long long totalNewlyDeactivated = 0;
  long long lastActiveZBins[4];
  long long lastDeactivatedZBins[4];
  long long lastActiveBefore = 0;
  long long lastActiveAfter = 0;
  double lastMaxActiveZ = -1.0;
  double lastMeanActiveZ = -1.0;

  GpuDiagOverall() { reset(); }

  void reset() {
    for (int i = 0; i < 32; ++i) totalExitCounts[i] = 0;
    for (int i = 0; i < 4; ++i) {
      lastActiveZBins[i] = 0;
      lastDeactivatedZBins[i] = 0;
    }
    totalWindows = 0;
    totalCalls = 0;
    totalDt = 0.0;
    totalNewlyDeactivated = 0;
    lastActiveBefore = 0;
    lastActiveAfter = 0;
    lastMaxActiveZ = -1.0;
    lastMeanActiveZ = -1.0;
  }

  void absorbWindow(const GpuDiagWindow &diag) {
    totalWindows += 1;
    totalCalls += diag.callCount;
    totalDt += diag.windowDt;
    totalNewlyDeactivated += diag.newlyDeactivated;
    for (int i = 0; i < 32; ++i) totalExitCounts[i] += diag.exitCounts[i];
    for (int i = 0; i < 4; ++i) {
      lastActiveZBins[i] = diag.activeZBins[i];
      lastDeactivatedZBins[i] = diag.deactivatedZBins[i];
    }
    lastActiveBefore = diag.activeBeforeLast;
    lastActiveAfter = diag.activeAfterLast;
    lastMaxActiveZ = diag.maxActiveZLast;
    lastMeanActiveZ = diag.meanActiveZLast;
  }
};

static const char* gpu_diag_summary_filename() {
  return "gpu_diag_summary.txt";
}

static void gpu_diag_init_summary_file_once() {
  static bool initialized = false;
  if (initialized) return;
  initialized = true;

  std::ofstream ofs(gpu_diag_summary_filename(), std::ios::out | std::ios::trunc);
  ofs << "GPU plume diagnostic summary\n";
  ofs << "=============================================\n";
  ofs << "This file is generated automatically by AdvectParticleGPU_demo_diag_lastwindow.cu\n";
  ofs << "Each block corresponds to one 60 s diagnostic window.\n\n";
}

static void gpu_diag_append_summary_file(
    const GpuDiagWindow &diag,
    const GpuDiagOverall &overall,
    const double zStart,
    const double zEnd
) {
  gpu_diag_init_summary_file_once();
  std::ofstream ofs(gpu_diag_summary_filename(), std::ios::out | std::ios::app);
  if (!ofs) return;

  auto pct = [](long long a, long long b) -> double {
    return (b > 0) ? (100.0 * (double)a / (double)b) : 0.0;
  };

  ofs << "[WINDOW] =============================================\n";
  ofs << "window_index=" << overall.totalWindows
      << "  steps_in_window=" << diag.callCount
      << "  dt_sum=" << diag.windowDt << " s\n";
  ofs << "newly_deactivated=" << diag.newlyDeactivated << "\n";
  ofs << "exit_reason_counts(window):\n";
  for (int code = 1; code <= GPU_EXIT_CODE_MAX; ++code) {
    if (diag.exitCounts[code] > 0) {
      ofs << "  code=" << code << "  "
          << gpu_exit_reason_name((unsigned char)code)
          << "  count=" << diag.exitCounts[code]
          << "  pct=" << std::fixed << std::setprecision(3)
          << pct(diag.exitCounts[code], diag.newlyDeactivated) << "%\n";
    }
  }
  ofs << std::defaultfloat;

  ofs << "active_last_step: before=" << diag.activeBeforeLast
      << " after=" << diag.activeAfterLast
      << " maxActiveZ=" << diag.maxActiveZLast
      << " meanActiveZ=" << diag.meanActiveZLast << "\n";

  ofs << "active_z_bins_last_step: "
      << "[0-25%]=" << diag.activeZBins[0] << ", "
      << "[25-50%]=" << diag.activeZBins[1] << ", "
      << "[50-75%]=" << diag.activeZBins[2] << ", "
      << "[75-100%]=" << diag.activeZBins[3] << "\n";
  ofs << "deactivated_z_bins(window): "
      << "[0-25%]=" << diag.deactivatedZBins[0] << ", "
      << "[25-50%]=" << diag.deactivatedZBins[1] << ", "
      << "[50-75%]=" << diag.deactivatedZBins[2] << ", "
      << "[75-100%]=" << diag.deactivatedZBins[3] << "\n";

  if (!diag.sampleLines.empty()) {
    ofs << "sample_particles(window, up to 8):\n";
    for (const auto &s : diag.sampleLines) {
      ofs << "  " << s << "\n";
    }
  }

  ofs << "cumulative_summary_until_this_window:\n";
  ofs << "  total_windows=" << overall.totalWindows
      << "  total_steps=" << overall.totalCalls
      << "  total_dt=" << overall.totalDt << " s\n";
  ofs << "  total_newly_deactivated=" << overall.totalNewlyDeactivated << "\n";
  ofs << "  total_exit_reason_counts:\n";
  for (int code = 1; code <= GPU_EXIT_CODE_MAX; ++code) {
    if (overall.totalExitCounts[code] > 0) {
      ofs << "    code=" << code << "  "
          << gpu_exit_reason_name((unsigned char)code)
          << "  count=" << overall.totalExitCounts[code]
          << "  pct=" << std::fixed << std::setprecision(3)
          << pct(overall.totalExitCounts[code], overall.totalNewlyDeactivated) << "%\n";
    }
  }
  ofs << std::defaultfloat;
  ofs << "  last_active_z_bins="
      << " [0-25%]=" << overall.lastActiveZBins[0]
      << " [25-50%]=" << overall.lastActiveZBins[1]
      << " [50-75%]=" << overall.lastActiveZBins[2]
      << " [75-100%]=" << overall.lastActiveZBins[3] << "\n";
  ofs << "  last_deactivated_z_bins="
      << " [0-25%]=" << overall.lastDeactivatedZBins[0]
      << " [25-50%]=" << overall.lastDeactivatedZBins[1]
      << " [50-75%]=" << overall.lastDeactivatedZBins[2]
      << " [75-100%]=" << overall.lastDeactivatedZBins[3] << "\n";
  ofs << "  z_range_for_bins=[" << zStart << ", " << zEnd << "]\n";
  ofs << "=======================================================\n\n";
}

static int gpu_diag_bin_from_z(
    const double z,
    const double zStart,
    const double zEnd
) {
  if (zEnd <= zStart) return 0;
  double a = (z - zStart) / (zEnd - zStart);
  if (a < 0.25) return 0;
  if (a < 0.50) return 1;
  if (a < 0.75) return 2;
  return 3;
}

static void gpu_print_diag_window(
    const GpuDiagWindow &diag,
    const double zStart,
    const double zEnd
) {
  std::cout << "[GPU-DIAG] =============================================\n";
  std::cout << "[GPU-DIAG] 最近一个时间窗口: " << diag.callCount
            << " 个时间步, 累计 dt = " << diag.windowDt << " s\n";
  std::cout << "[GPU-DIAG] 本窗口新失活粒子数 = " << diag.newlyDeactivated << "\n";
  std::cout << "[GPU-DIAG] 本窗口失活原因统计:\n";
  for (int code = 1; code <= GPU_EXIT_CODE_MAX; ++code) {
    if (diag.exitCounts[code] > 0) {
      std::cout << "  code=" << code << "  "
                << gpu_exit_reason_name((unsigned char)code)
                << " : " << diag.exitCounts[code] << "\n";
    }
  }

  std::cout << "[GPU-DIAG] 当前步 active: before=" << diag.activeBeforeLast
            << " after=" << diag.activeAfterLast
            << "  maxActiveZ=" << diag.maxActiveZLast
            << "  meanActiveZ=" << diag.meanActiveZLast << "\n";

  std::cout << "[GPU-DIAG] 当前步 active 粒子竖向分布:\n";
  std::cout << "  0%-25%   : " << diag.activeZBins[0] << "\n";
  std::cout << "  25%-50%  : " << diag.activeZBins[1] << "\n";
  std::cout << "  50%-75%  : " << diag.activeZBins[2] << "\n";
  std::cout << "  75%-100% : " << diag.activeZBins[3] << "\n";

  std::cout << "[GPU-DIAG] 本窗口失活粒子最终 z 分布:\n";
  std::cout << "  0%-25%   : " << diag.deactivatedZBins[0] << "\n";
  std::cout << "  25%-50%  : " << diag.deactivatedZBins[1] << "\n";
  std::cout << "  50%-75%  : " << diag.deactivatedZBins[2] << "\n";
  std::cout << "  75%-100% : " << diag.deactivatedZBins[3] << "\n";

  if (!diag.sampleLines.empty()) {
    std::cout << "[GPU-DIAG] 当前窗口样本粒子(最多8条):\n";
    for (const auto &s : diag.sampleLines) {
      std::cout << "  " << s << "\n";
    }
  }
  std::cout << "[GPU-DIAG] =============================================\n";
}

static void uploadParticleRangeToDevice(
    const std::vector<Particle*>& particles,
    int iBeg,
    int iEnd
) {
  if (iEnd <= iBeg) return;

  const double rhoAir = 1.225;
  const double nuAir  = 1.506e-5;

  const int cnt = iEnd - iBeg;

  std::vector<double> hx(cnt), hy(cnt), hz(cnt);
  std::vector<double> hdisX(cnt), hdisY(cnt), hdisZ(cnt);
  std::vector<double> huMean(cnt), hvMean(cnt), hwMean(cnt);
  std::vector<double> huFluct(cnt), hvFluct(cnt), hwFluct(cnt);
  std::vector<double> huFluct_old(cnt), hvFluct_old(cnt), hwFluct_old(cnt);
  std::vector<double> hdelta_uFluct(cnt), hdelta_vFluct(cnt), hdelta_wFluct(cnt);
  std::vector<double> htxx_old(cnt), htxy_old(cnt), htxz_old(cnt);
  std::vector<double> htyy_old(cnt), htyz_old(cnt), htzz_old(cnt);
  std::vector<double> hCoEps(cnt), hvs(cnt);
  std::vector<unsigned char> hActive(cnt), hRogue(cnt);
  std::vector<double> hm(cnt), hm_kg(cnt), hwdecay(cnt);
  std::vector<double> hrho(cnt), hd_m(cnt), hc1(cnt), hc2(cnt);
  std::vector<unsigned char> hDepFlag(cnt);

  for (int k = 0; k < cnt; ++k) {
    Particle* p = particles[iBeg + k];

    // 只对本次新增/上传的粒子算一次 settling velocity
    p->setSettlingVelocity(rhoAir, nuAir);

    hx[k] = p->xPos; hy[k] = p->yPos; hz[k] = p->zPos;
    hdisX[k] = p->disX; hdisY[k] = p->disY; hdisZ[k] = p->disZ;

    huMean[k] = p->uMean; hvMean[k] = p->vMean; hwMean[k] = p->wMean;
    huFluct[k] = p->uFluct; hvFluct[k] = p->vFluct; hwFluct[k] = p->wFluct;

    huFluct_old[k] = p->uFluct_old;
    hvFluct_old[k] = p->vFluct_old;
    hwFluct_old[k] = p->wFluct_old;

    hdelta_uFluct[k] = p->delta_uFluct;
    hdelta_vFluct[k] = p->delta_vFluct;
    hdelta_wFluct[k] = p->delta_wFluct;

    htxx_old[k] = p->txx_old;
    htxy_old[k] = p->txy_old;
    htxz_old[k] = p->txz_old;
    htyy_old[k] = p->tyy_old;
    htyz_old[k] = p->tyz_old;
    htzz_old[k] = p->tzz_old;

    hCoEps[k] = (p->CoEps > 1.0e-12) ? p->CoEps : 1.0e-6;
    hvs[k] = p->vs;
	
    hm[k]      = p->m;
    hm_kg[k]   = p->m_kg;
    hwdecay[k] = p->wdecay;

    hrho[k] = p->rho;
    hd_m[k] = p->d_m;
    hc1[k]  = p->c1;
    hc2[k]  = p->c2;

    hDepFlag[k] = p->depFlag ? 1 : 0;
    hActive[k] = p->isActive ? 1 : 0;
    hRogue[k]  = p->isRogue  ? 1 : 0;
  }

  cudaCheck(cudaMemcpy(gPart.dx_ptr + iBeg, hx.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D x");
  cudaCheck(cudaMemcpy(gPart.dy_ptr + iBeg, hy.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D y");
  cudaCheck(cudaMemcpy(gPart.dz_ptr + iBeg, hz.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D z");

  cudaCheck(cudaMemcpy(gPart.ddisX + iBeg, hdisX.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D disX");
  cudaCheck(cudaMemcpy(gPart.ddisY + iBeg, hdisY.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D disY");
  cudaCheck(cudaMemcpy(gPart.ddisZ + iBeg, hdisZ.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D disZ");

  cudaCheck(cudaMemcpy(gPart.duMean + iBeg, huMean.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D uMean");
  cudaCheck(cudaMemcpy(gPart.dvMean + iBeg, hvMean.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D vMean");
  cudaCheck(cudaMemcpy(gPart.dwMean + iBeg, hwMean.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D wMean");

  cudaCheck(cudaMemcpy(gPart.duFluct + iBeg, huFluct.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D uFluct");
  cudaCheck(cudaMemcpy(gPart.dvFluct + iBeg, hvFluct.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D vFluct");
  cudaCheck(cudaMemcpy(gPart.dwFluct + iBeg, hwFluct.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D wFluct");

  cudaCheck(cudaMemcpy(gPart.duFluct_old + iBeg, huFluct_old.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D uFluct_old");
  cudaCheck(cudaMemcpy(gPart.dvFluct_old + iBeg, hvFluct_old.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D vFluct_old");
  cudaCheck(cudaMemcpy(gPart.dwFluct_old + iBeg, hwFluct_old.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D wFluct_old");

  cudaCheck(cudaMemcpy(gPart.ddelta_uFluct + iBeg, hdelta_uFluct.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D delta_uFluct");
  cudaCheck(cudaMemcpy(gPart.ddelta_vFluct + iBeg, hdelta_vFluct.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D delta_vFluct");
  cudaCheck(cudaMemcpy(gPart.ddelta_wFluct + iBeg, hdelta_wFluct.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D delta_wFluct");

  cudaCheck(cudaMemcpy(gPart.dtxx_old + iBeg, htxx_old.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D txx_old");
  cudaCheck(cudaMemcpy(gPart.dtxy_old + iBeg, htxy_old.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D txy_old");
  cudaCheck(cudaMemcpy(gPart.dtxz_old + iBeg, htxz_old.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D txz_old");
  cudaCheck(cudaMemcpy(gPart.dtyy_old + iBeg, htyy_old.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D tyy_old");
  cudaCheck(cudaMemcpy(gPart.dtyz_old + iBeg, htyz_old.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D tyz_old");
  cudaCheck(cudaMemcpy(gPart.dtzz_old + iBeg, htzz_old.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D tzz_old");

  cudaCheck(cudaMemcpy(gPart.dCoEps + iBeg, hCoEps.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D CoEps");
  cudaCheck(cudaMemcpy(gPart.dvs + iBeg, hvs.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D vs");

  cudaCheck(cudaMemcpy(gPart.dm + iBeg,      hm.data(),      cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D m");
  cudaCheck(cudaMemcpy(gPart.dm_kg + iBeg,   hm_kg.data(),   cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D m_kg");
  cudaCheck(cudaMemcpy(gPart.dwdecay + iBeg, hwdecay.data(), cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D wdecay");

  cudaCheck(cudaMemcpy(gPart.drho + iBeg,  hrho.data(),  cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D rho");
  cudaCheck(cudaMemcpy(gPart.dd_m + iBeg,  hd_m.data(),  cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D d_m");
  cudaCheck(cudaMemcpy(gPart.dc1 + iBeg,   hc1.data(),   cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D c1");
  cudaCheck(cudaMemcpy(gPart.dc2 + iBeg,   hc2.data(),   cnt * sizeof(double), cudaMemcpyHostToDevice), "range H2D c2");

  cudaCheck(cudaMemcpy(gPart.dDepFlag + iBeg, hDepFlag.data(), cnt * sizeof(unsigned char), cudaMemcpyHostToDevice), "range H2D depFlag");

  cudaCheck(cudaMemcpy(gPart.dActive + iBeg, hActive.data(), cnt * sizeof(unsigned char), cudaMemcpyHostToDevice), "range H2D active");
  cudaCheck(cudaMemcpy(gPart.dRogue + iBeg, hRogue.data(), cnt * sizeof(unsigned char), cudaMemcpyHostToDevice), "range H2D rogue");

  cudaCheck(cudaMemset(gPart.dExitCode + iBeg, 0, cnt * sizeof(unsigned char)), "range memset exitCode");
}

static void ensureParticlesResidentOnDevice(const std::vector<Particle*>& particles) {
  const int n = static_cast<int>(particles.size());
  if (n == 0) return;

  ensureParticleCache(n);

  // 第一次上传：全量
  if (!gPart.uploaded) {
    uploadParticleRangeToDevice(particles, 0, n);
    gPart.n_used = n;
    gPart.uploaded = true;
    gPart.hostDirty = false;
    return;
  }

  // 如果 host 粒子数变小了，说明外部做了重组/压缩，这时直接全量重传当前 host 粒子
  if (n < gPart.n_used) {
    uploadParticleRangeToDevice(particles, 0, n);
    gPart.n_used = n;
    gPart.hostDirty = false;
    return;
  }

  // 如果新增了粒子，只上传新增区间
  if (n > gPart.n_used) {
    uploadParticleRangeToDevice(particles, gPart.n_used, n);
    gPart.n_used = n;
  }
}

void advectParticlesGPU_demo_syncFlagsToHost(const std::vector<Particle*>& particles) {
  const int n = static_cast<int>(particles.size());
  if (n == 0 || !gPart.uploaded) return;

  std::vector<unsigned char> hActive(n), hRogue(n);

  cudaCheck(cudaMemcpy(hActive.data(), gPart.dActive, n * sizeof(unsigned char), cudaMemcpyDeviceToHost),
            "syncFlags D2H active");
  cudaCheck(cudaMemcpy(hRogue.data(), gPart.dRogue, n * sizeof(unsigned char), cudaMemcpyDeviceToHost),
            "syncFlags D2H rogue");

  for (int i = 0; i < n; ++i) {
    Particle* p = particles[i];
    p->isActive = (hActive[i] != 0);
    p->isRogue  = (hRogue[i]  != 0);
  }
}

void advectParticlesGPU_demo_syncOutputFieldsToHost(const std::vector<Particle*>& particles) {
  const int n = static_cast<int>(particles.size());
  if (n == 0 || !gPart.uploaded) return;

  std::vector<double> hx(n), hy(n), hz(n);
  std::vector<double> hm(n), hm_kg(n);
  std::vector<unsigned char> hActive(n), hRogue(n);

  cudaCheck(cudaMemcpy(hx.data(), gPart.dx_ptr, n * sizeof(double), cudaMemcpyDeviceToHost),
            "syncOutput D2H x");
  cudaCheck(cudaMemcpy(hy.data(), gPart.dy_ptr, n * sizeof(double), cudaMemcpyDeviceToHost),
            "syncOutput D2H y");
  cudaCheck(cudaMemcpy(hz.data(), gPart.dz_ptr, n * sizeof(double), cudaMemcpyDeviceToHost),
            "syncOutput D2H z");

  cudaCheck(cudaMemcpy(hm.data(), gPart.dm, n * sizeof(double), cudaMemcpyDeviceToHost),
            "syncOutput D2H m");
  cudaCheck(cudaMemcpy(hm_kg.data(), gPart.dm_kg, n * sizeof(double), cudaMemcpyDeviceToHost),
            "syncOutput D2H m_kg");

  cudaCheck(cudaMemcpy(hActive.data(), gPart.dActive, n * sizeof(unsigned char), cudaMemcpyDeviceToHost),
            "syncOutput D2H active");
  cudaCheck(cudaMemcpy(hRogue.data(), gPart.dRogue, n * sizeof(unsigned char), cudaMemcpyDeviceToHost),
            "syncOutput D2H rogue");

  for (int i = 0; i < n; ++i) {
    Particle* p = particles[i];
    p->xPos = hx[i];
    p->yPos = hy[i];
    p->zPos = hz[i];

    p->m    = hm[i];
    p->m_kg = hm_kg[i];

    p->isActive = (hActive[i] != 0);
    p->isRogue  = (hRogue[i]  != 0);
  }

  // 注意：这里不要把 gPart.hostDirty = false;
  // 因为这只是“输出所需字段”的轻量同步，
  // uMean / uFluct / delta / txx_old ... 这些完整字段在 host 端仍然不是最新的。
}

void advectParticlesGPU_demo_syncAllToHost(const std::vector<Particle*>& particles) {
  const int n = static_cast<int>(particles.size());
  if (n == 0 || !gPart.uploaded) return;

  std::vector<double> hx(n), hy(n), hz(n);
  std::vector<double> hdisX(n), hdisY(n), hdisZ(n);
  std::vector<double> huMean(n), hvMean(n), hwMean(n);
  std::vector<double> huFluct(n), hvFluct(n), hwFluct(n);
  std::vector<double> huFluct_old(n), hvFluct_old(n), hwFluct_old(n);
  std::vector<double> hdelta_uFluct(n), hdelta_vFluct(n), hdelta_wFluct(n);
  std::vector<double> htxx_old(n), htxy_old(n), htxz_old(n);
  std::vector<double> htyy_old(n), htyz_old(n), htzz_old(n);
  std::vector<double> hCoEps(n);
  std::vector<unsigned char> hActive(n), hRogue(n);
  std::vector<double> hm(n), hm_kg(n);


  cudaCheck(cudaMemcpy(hx.data(), gPart.dx_ptr, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H x");
  cudaCheck(cudaMemcpy(hy.data(), gPart.dy_ptr, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H y");
  cudaCheck(cudaMemcpy(hz.data(), gPart.dz_ptr, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H z");

  cudaCheck(cudaMemcpy(hdisX.data(), gPart.ddisX, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H disX");
  cudaCheck(cudaMemcpy(hdisY.data(), gPart.ddisY, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H disY");
  cudaCheck(cudaMemcpy(hdisZ.data(), gPart.ddisZ, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H disZ");

  cudaCheck(cudaMemcpy(huMean.data(), gPart.duMean, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H uMean");
  cudaCheck(cudaMemcpy(hvMean.data(), gPart.dvMean, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H vMean");
  cudaCheck(cudaMemcpy(hwMean.data(), gPart.dwMean, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H wMean");

  cudaCheck(cudaMemcpy(huFluct.data(), gPart.duFluct, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H uFluct");
  cudaCheck(cudaMemcpy(hvFluct.data(), gPart.dvFluct, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H vFluct");
  cudaCheck(cudaMemcpy(hwFluct.data(), gPart.dwFluct, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H wFluct");

  cudaCheck(cudaMemcpy(huFluct_old.data(), gPart.duFluct_old, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H uFluct_old");
  cudaCheck(cudaMemcpy(hvFluct_old.data(), gPart.dvFluct_old, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H vFluct_old");
  cudaCheck(cudaMemcpy(hwFluct_old.data(), gPart.dwFluct_old, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H wFluct_old");

  cudaCheck(cudaMemcpy(hdelta_uFluct.data(), gPart.ddelta_uFluct, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H delta_uFluct");
  cudaCheck(cudaMemcpy(hdelta_vFluct.data(), gPart.ddelta_vFluct, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H delta_vFluct");
  cudaCheck(cudaMemcpy(hdelta_wFluct.data(), gPart.ddelta_wFluct, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H delta_wFluct");

  cudaCheck(cudaMemcpy(htxx_old.data(), gPart.dtxx_old, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H txx_old");
  cudaCheck(cudaMemcpy(htxy_old.data(), gPart.dtxy_old, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H txy_old");
  cudaCheck(cudaMemcpy(htxz_old.data(), gPart.dtxz_old, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H txz_old");
  cudaCheck(cudaMemcpy(htyy_old.data(), gPart.dtyy_old, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H tyy_old");
  cudaCheck(cudaMemcpy(htyz_old.data(), gPart.dtyz_old, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H tyz_old");
  cudaCheck(cudaMemcpy(htzz_old.data(), gPart.dtzz_old, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H tzz_old");

  cudaCheck(cudaMemcpy(hCoEps.data(), gPart.dCoEps, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H CoEps");
  
  cudaCheck(cudaMemcpy(hm.data(),    gPart.dm,    n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H m");
  cudaCheck(cudaMemcpy(hm_kg.data(), gPart.dm_kg, n * sizeof(double), cudaMemcpyDeviceToHost), "syncAll D2H m_kg");
  
  cudaCheck(cudaMemcpy(hActive.data(), gPart.dActive, n * sizeof(unsigned char), cudaMemcpyDeviceToHost), "syncAll D2H active");
  cudaCheck(cudaMemcpy(hRogue.data(), gPart.dRogue, n * sizeof(unsigned char), cudaMemcpyDeviceToHost), "syncAll D2H rogue");

  for (int i = 0; i < n; ++i) {
    Particle* p = particles[i];
    p->xPos = hx[i]; p->yPos = hy[i]; p->zPos = hz[i];
    p->disX = hdisX[i]; p->disY = hdisY[i]; p->disZ = hdisZ[i];

    p->uMean = huMean[i]; p->vMean = hvMean[i]; p->wMean = hwMean[i];
    p->uFluct = huFluct[i]; p->vFluct = hvFluct[i]; p->wFluct = hwFluct[i];

    p->uFluct_old = huFluct_old[i];
    p->vFluct_old = hvFluct_old[i];
    p->wFluct_old = hwFluct_old[i];

    p->delta_uFluct = hdelta_uFluct[i];
    p->delta_vFluct = hdelta_vFluct[i];
    p->delta_wFluct = hdelta_wFluct[i];

    p->txx_old = htxx_old[i];
    p->txy_old = htxy_old[i];
    p->txz_old = htxz_old[i];
    p->tyy_old = htyy_old[i];
    p->tyz_old = htyz_old[i];
    p->tzz_old = htzz_old[i];

    p->m    = hm[i];
    p->m_kg = hm_kg[i];

    p->CoEps = hCoEps[i];
    p->isActive = (hActive[i] != 0);
    p->isRogue  = (hRogue[i]  != 0);
  }

  gPart.hostDirty = false;
}

void advectParticlesGPU_demo_flushDepositionToHost(Deposition* deposition) {
  if (!deposition) return;
  if (!gGrid.ready) return;
  if (!gGrid.dDepMass) return;

  size_t n = gGrid.sz_dep;
  if (deposition->depcvol.size() < n) {
    n = deposition->depcvol.size();
  }
  if (n == 0) return;

  // 复用 host buffer，避免每次 flush 都重新分配
  if (gHostDepBuffer.size() < n) {
    gHostDepBuffer.resize(n);
  }

  cudaCheck(cudaMemcpy(gHostDepBuffer.data(),
                       gGrid.dDepMass,
                       n * sizeof(double),
                       cudaMemcpyDeviceToHost),
            "sync deposition grid D2H");

  for (size_t i = 0; i < n; ++i) {
    deposition->depcvol[i] += gHostDepBuffer[i];
  }

  cudaCheck(cudaMemset(gGrid.dDepMass, 0, n * sizeof(double)),
            "clear device deposition grid");
}

void advectParticlesGPU_demo_prepareEulerianBoxes(
    int nBoxesX, int nBoxesY, int nBoxesZ,
    float lBndx, float lBndy, float lBndz,
    float boxSizeX, float boxSizeY, float boxSizeZ)
{
  const int nCells = nBoxesX * nBoxesY * nBoxesZ;

  if (gBox.ready &&
      gBox.nBoxesX == nBoxesX &&
      gBox.nBoxesY == nBoxesY &&
      gBox.nBoxesZ == nBoxesZ &&
      gBox.lBndx == lBndx &&
      gBox.lBndy == lBndy &&
      gBox.lBndz == lBndz &&
      gBox.boxSizeX == boxSizeX &&
      gBox.boxSizeY == boxSizeY &&
      gBox.boxSizeZ == boxSizeZ) {
    return;
  }

  freeEulerianBoxCache();

  gBox.nBoxesX = nBoxesX;
  gBox.nBoxesY = nBoxesY;
  gBox.nBoxesZ = nBoxesZ;
  gBox.nCells  = nCells;

  gBox.lBndx = lBndx;
  gBox.lBndy = lBndy;
  gBox.lBndz = lBndz;

  gBox.boxSizeX = boxSizeX;
  gBox.boxSizeY = boxSizeY;
  gBox.boxSizeZ = boxSizeZ;

  cudaCheck(cudaMalloc(&gBox.dPBox, nCells * sizeof(int)), "malloc eulerian dPBox");
  cudaCheck(cudaMalloc(&gBox.dConc, nCells * sizeof(float)), "malloc eulerian dConc");

  cudaCheck(cudaMemset(gBox.dPBox, 0, nCells * sizeof(int)), "memset eulerian dPBox");
  cudaCheck(cudaMemset(gBox.dConc, 0, nCells * sizeof(float)), "memset eulerian dConc");

  gBox.ready = true;
}

void advectParticlesGPU_demo_accumulateEulerianBoxes(double dt)
{
  if (!gPart.uploaded || !gBox.ready || gPart.n_used <= 0) return;

  const int n = gPart.n_used;
  const int threads = 256;
  const int blocks  = (n + threads - 1) / threads;

  accumulate_eulerian_boxes_kernel<<<blocks, threads>>>(
    n,
    gPart.dx_ptr,
    gPart.dy_ptr,
    gPart.dz_ptr,
    gPart.dm,
    gPart.dwdecay,
    gPart.dActive,
    gBox.nBoxesX,
    gBox.nBoxesY,
    gBox.nBoxesZ,
    gBox.lBndx,
    gBox.lBndy,
    gBox.lBndz,
    gBox.boxSizeX,
    gBox.boxSizeY,
    gBox.boxSizeZ,
    (float)dt,
    gBox.dPBox,
    gBox.dConc);

  cudaCheck(cudaGetLastError(), "accumulate_eulerian_boxes_kernel launch");
}

void advectParticlesGPU_demo_flushEulerianBoxesToHost(
    std::vector<int>& pBox,
    std::vector<float>& conc,
    bool reset_after_flush)
{
  if (!gBox.ready) return;

  pBox.resize(gBox.nCells);
  conc.resize(gBox.nCells);

  cudaCheck(cudaMemcpy(pBox.data(), gBox.dPBox,
                       gBox.nCells * sizeof(int),
                       cudaMemcpyDeviceToHost),
            "flushEulerian D2H pBox");

  cudaCheck(cudaMemcpy(conc.data(), gBox.dConc,
                       gBox.nCells * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "flushEulerian D2H conc");

  static int dbgFlushCount = 0;
  if (dbgFlushCount < 8) {
    long long totalP = 0;
    double totalConc = 0.0;
    int nonZeroBox = 0;

    for (int i = 0; i < gBox.nCells; ++i) {
      totalP += (long long)pBox[i];
      totalConc += (double)conc[i];
      if (pBox[i] > 0 || conc[i] > 0.0f) {
        nonZeroBox++;
      }
    }

    std::cout << "[GPU Euler] flush #" << dbgFlushCount
              << " totalPBox=" << totalP
              << " totalConcRaw=" << totalConc
              << " nonZeroBox=" << nonZeroBox
              << " nCells=" << gBox.nCells
              << std::endl;

    dbgFlushCount++;
  }

  if (reset_after_flush) {
    cudaCheck(cudaMemset(gBox.dPBox, 0, gBox.nCells * sizeof(int)),
              "resetEulerian dPBox");
    cudaCheck(cudaMemset(gBox.dConc, 0, gBox.nCells * sizeof(float)),
              "resetEulerian dConc");
  }
}

void advectParticlesGPU_demo_resetEulerianBoxes()
{
  if (!gBox.ready) return;

  cudaCheck(cudaMemset(gBox.dPBox, 0, gBox.nCells * sizeof(int)),
            "resetEulerian dPBox");
  cudaCheck(cudaMemset(gBox.dConc, 0, gBox.nCells * sizeof(float)),
            "resetEulerian dConc");
}

void advectParticlesGPU_demo_invalidateParticleCache() {
  // 不释放显存，只把“当前 device 粒子内容有效”的标志清掉
  // 这样下一步会按 host 当前粒子列表重新全量上传，重新建立一一对应关系
  gPart.uploaded = false;
  gPart.n_used = 0;
  gPart.hostDirty = false;
}

void advectParticlesGPU_demo(
    double dt,
    const std::vector<Particle*>& particles,
    double boxSizeZ,
    WINDSGeneralData* WGD,
    TURBGeneralData* TGD,
    bool use_const_wind,
    double u0, double v0, double w0,
    bool clampZ_and_deactivate,
    float gaussian_scale,
    double courantNum,
    double dxy,
    double dz,
    double sim_dt,
    double vel_threshold,
    int bcTypeX,
    int bcTypeY,
    int bcTypeZ
) {
  static bool printed = false;
  if (!printed) {
    printed = true;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
      std::cout << "[GPU STAGE1-PERF] Using GPU: " << prop.name << std::endl;
    }
  }

  const int n = static_cast<int>(particles.size());
  if (n == 0) return;

  if ((!WGD || !TGD) && !use_const_wind) {
    throw std::runtime_error("[GPU STAGE1-PERF] WGD/TGD is null while use_const_wind=false");
  }

  ensureParticleCache(n);
  if (!use_const_wind) ensureGridCache(WGD, TGD);

   // 关键：只在第一次 / 新增粒子时上传
  ensureParticlesResidentOnDevice(particles);

  ensureStepStats();
  cudaCheck(cudaMemset(gStep.dActiveCount,   0, sizeof(int)), "memset step active");
  cudaCheck(cudaMemset(gStep.dInactiveCount, 0, sizeof(int)), "memset step inactive");
  cudaCheck(cudaMemset(gStep.dRogueCount,    0, sizeof(int)), "memset step rogue");
  
  int nx = WGD ? WGD->nx : 0;
  int ny = WGD ? WGD->ny : 0;
  int nz = WGD ? WGD->nz : 0;

  double dx = WGD ? WGD->dx : 1.0;
  double dy = WGD ? WGD->dy : 1.0;

  double xStart = WGD ? (WGD->x[1] - 0.5 * dx) : 0.0;
  double xEnd   = WGD ? (WGD->x[nx - 3] + 0.5 * dx) : 0.0;
  double yStart = WGD ? (WGD->y[1] - 0.5 * dy) : 0.0;
  double yEnd   = WGD ? (WGD->y[ny - 3] + 0.5 * dy) : 0.0;
  double zStart = WGD ? WGD->z_face[1] : 0.0;
  double zEnd   = WGD ? WGD->z_face[nz - 2] : boxSizeZ;

  double xBCStart = WGD ? (WGD->x.front() - 0.5 * dx) : 0.0;
  double xBCEnd   = WGD ? (WGD->x.back()  + 0.5 * dx) : 0.0;
  double yBCStart = WGD ? (WGD->y.front() - 0.5 * dy) : 0.0;
  double yBCEnd   = WGD ? (WGD->y.back()  + 0.5 * dy) : 0.0;
  double zBCStart = WGD ? WGD->z_face.front() : 0.0;
  double zBCEnd   = WGD ? WGD->z_face.back()  : boxSizeZ;
  double domainZstart = zBCStart;

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  advect_langevin_kernel<<<blocks, threads>>>(
      n, dt,
      gPart.dx_ptr, gPart.dy_ptr, gPart.dz_ptr,
      gPart.ddisX, gPart.ddisY, gPart.ddisZ,
      gPart.duMean, gPart.dvMean, gPart.dwMean,
      gPart.duFluct, gPart.dvFluct, gPart.dwFluct,
      gPart.duFluct_old, gPart.dvFluct_old, gPart.dwFluct_old,
      gPart.ddelta_uFluct, gPart.ddelta_vFluct, gPart.ddelta_wFluct,
      gPart.dtxx_old, gPart.dtxy_old, gPart.dtxz_old, gPart.dtyy_old, gPart.dtyz_old, gPart.dtzz_old,
      gPart.dCoEps, gPart.dvs,
      gPart.dm, gPart.dm_kg,
      gPart.drho, gPart.dd_m, gPart.dc1, gPart.dc2,
      gPart.dDepFlag,
      gPart.dActive, gPart.dRogue, gPart.dExitCode,
      gStep.dActiveCount, gStep.dInactiveCount, gStep.dRogueCount,
      boxSizeZ,
      clampZ_and_deactivate ? 1 : 0,
      use_const_wind ? 1 : 0,
      u0, v0, w0,
      gaussian_scale,
      courantNum,
      dxy,
      dz,
      sim_dt,
      vel_threshold,
      gGrid.dWGD_u, gGrid.dWGD_v, gGrid.dWGD_w,
      gGrid.dIcellflag, gGrid.dXcenters, gGrid.dYcenters,
      gGrid.dTGD_CoEps,
      gGrid.dTGD_txx, gGrid.dTGD_txy, gGrid.dTGD_txz,
      gGrid.dTGD_tyy, gGrid.dTGD_tyz, gGrid.dTGD_tzz,
      gGrid.dTGD_div_tau_x, gGrid.dTGD_div_tau_y, gGrid.dTGD_div_tau_z,
      gGrid.dTGD_nuT,
      gGrid.dMixingLengths,
      gGrid.dz_nodes, gGrid.dz_faces,
      gGrid.dDepMass,
      nx, ny, nz,
      dx, dy,
      xStart, xEnd, yStart, yEnd, zStart, zEnd,
      bcTypeX, bcTypeY, bcTypeZ,
      xBCStart, xBCEnd, yBCStart, yBCEnd, zBCStart, zBCEnd,
      domainZstart,
      gPart.dStates
  );


  cudaCheck(cudaGetLastError(), "kernel launch");

  // 不回传整批 flags，只回传 3 个计数
  cudaCheck(cudaMemcpy(&gStep.hActiveCount,
                       gStep.dActiveCount,
                       sizeof(int),
                       cudaMemcpyDeviceToHost),
            "step active D2H");

  cudaCheck(cudaMemcpy(&gStep.hInactiveCount,
                       gStep.dInactiveCount,
                       sizeof(int),
                       cudaMemcpyDeviceToHost),
            "step inactive D2H");

  cudaCheck(cudaMemcpy(&gStep.hRogueCount,
                       gStep.dRogueCount,
                       sizeof(int),
                       cudaMemcpyDeviceToHost),
            "step rogue D2H");

  static std::vector<unsigned char> hExitCodes;
  static long long windowCounts[32] = {0};
  static int windowCalls = 0;
  static double windowDt = 0.0;

  hExitCodes.resize(n);
  cudaCheck(cudaMemcpy(hExitCodes.data(),
                       gPart.dExitCode,
                       n * sizeof(unsigned char),
                       cudaMemcpyDeviceToHost),
            "step exitCode D2H");

  for (int ii = 0; ii < n; ++ii) {
    unsigned char code = hExitCodes[ii];
    if (code > 0 && code < 32) {
      windowCounts[code] += 1;
    }
  }

  windowCalls += 1;
  windowDt += dt;

  const bool shouldPrintDiag = (windowDt >= 60.0 - 1.0e-9);
  if (shouldPrintDiag) {
    gpu_print_exit_window_summary(windowCounts, windowCalls, windowDt, gStep.hActiveCount);
    for (int code = 0; code < 32; ++code) windowCounts[code] = 0;
    windowCalls = 0;
    windowDt = 0.0;
  }

  gPart.hostDirty = true;
}