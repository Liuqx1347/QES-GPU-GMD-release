# GPU-Accelerated QES-Turb and QES-Plume in QES v2.0.2

This repository contains the GPU-enabled QES source code and openly distributable model test cases associated with the manuscript:

**GPU-Accelerated QES-Turb and QES-Plume in QES v2.0.2 for Rapid High-Resolution Atmospheric Dispersion Modeling in Complex Nuclear Power Plant Environments**

The repository is prepared as the frozen model-code release accompanying submission to *Geoscientific Model Development (GMD)*.

## 1. Overview

Quick Environmental Simulation (QES) is a fast-response atmospheric modeling framework for wind-field, turbulence, and pollutant-dispersion simulations in complex terrain and built environments.

The computational chain considered in this study consists of:

- **QES-Winds**: generation of the three-dimensional mean wind field;
- **QES-Turb**: calculation of turbulence quantities required by the dispersion model;
- **QES-Plume**: Lagrangian stochastic particle transport, concentration sampling, and deposition calculations.

This release is based on **QES v2.0.2** and contains the GPU developments described in the accompanying manuscript. The principal developments are:

1. GPU acceleration of the QES-Turb wall-below local mixing-length calculation using a target-cell parallel strategy;
2. GPU acceleration of QES-Plume particle calculations, including particle advection, concentration statistics, and deposition accumulation;
3. GPU-oriented particle-data organization and reduced CPU-GPU data-transfer and synchronization overhead.

The original physical formulations of QES are retained. The public archive provides the GPU-enabled source implementation together with openly distributable QES test cases that can be used to compile and execute representative model workflows.

## 2. Repository structure

```text
QES-GPU-GMD-release/
├── README.md
├── BUILD_ENVIRONMENT.txt
├── LICENSE
├── MODIFIED_FILES.txt
└── code/
    ├── CMakeLists.txt
    ├── docs/
    ├── qes/
    ├── qesWinds/
    ├── qesPlume/
    ├── src/
    └── testCases/
    └── EPA_7x11array/
```

### Main contents

- `code/` — QES v2.0.2 source code used in this study.
- `code/src/winds/` — QES-Winds and QES-Turb source files, including the GPU implementation of the wall-below local mixing-length calculation.
- `code/src/plume/` — QES-Plume source files, including the GPU particle calculations.
- `code/testCases/` — openly distributable QES model test cases.
- `code/testCases/EPA_7x11array/` — representative public building-array case used to check execution of both the QES-Turb wall-below GPU path and the QES-Plume GPU particle path.
- `BUILD_ENVIRONMENT.txt` — reference software and hardware environment used for the archived build.
- `MODIFIED_FILES.txt` — summary of source files modified for the GPU developments in this study.
- `LICENSE` — software license for this archived release.

Generated build products and model outputs are not part of the frozen source archive.

## 3. Reference computational environment

The principal GPU implementation and performance evaluation reported in the manuscript used the following reference platform:

- Operating system: Ubuntu 20.04.6 LTS
- CPU: Intel Xeon E5-2680 v4
- Memory: approximately 62 GiB
- GPU: NVIDIA GeForce RTX 3090, 24 GB
- CUDA Toolkit: 12.4
- GCC: 8.4
- CMake: 3.28.6
- OpenMP: 4.5
- Boost: 1.77
- NetCDF4 / NetCDF-C++4
- GDAL
- libpng

Additional hardware evaluation reported in the manuscript was performed on an NVIDIA GeForce RTX 4090 platform.

For the archived reference environment, see:

```text
BUILD_ENVIRONMENT.txt
```

## 4. Software requirements

A Linux environment is recommended.

The GPU-enabled build requires:

- CUDA-capable NVIDIA GPU
- NVIDIA CUDA Toolkit
- C and C++ compilers compatible with the installed CUDA Toolkit
- CMake 3.18 or newer
- Boost
- NetCDF and NetCDF-C++4
- GDAL
- libpng
- OpenMP support

The archived source identifies the model as:

```text
QES v2.0.2
```

OptiX is not required for the GPU developments evaluated in this study. CMake may report that OptiX is unavailable; this does not prevent the CUDA-enabled QES-Turb and QES-Plume implementations in this release from being built.

## 5. Clean build

The source should be compiled using an **out-of-source build**.

From the repository root:

```bash
cmake -S code -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DHAS_CUDA_SUPPORT=ON
```

Compile with:

```bash
cmake --build build -j4
```

The number of parallel build jobs may be changed for the local machine.

A successful build generates:

```text
build/qes/qes
build/qesWinds/qesWinds
build/qesPlume/qesPlume
```

The archived source was successfully configured and compiled from a clean build directory with CUDA support enabled.

### Important

Do not reuse `CMakeCache.txt`, `CMakeFiles/`, `Makefile`, or other generated build files from another machine or source-tree location. Generated CMake files may contain machine-specific absolute paths.

## 6. Representative public GPU test case: EPA 7×11 building array

The public EPA 7×11 building-array case is retained under:

```text
code/testCases/EPA_7x11array/
```

It contains a regular building-array geometry, QES-Winds/QES-Turb input, QES-Plume input, and MATLAB post-processing utilities.

For GPU functional verification, the QES-Turb local mixing-length method is set to the serial local-mixing formulation so that the GPU wall-below implementation is exercised.

The case is intended as a **functional GPU verification case**. It is not intended to reproduce the full site-specific benchmark configuration or the exact performance values reported in the manuscript.

### 6.1 QES-Winds and QES-Turb

From the EPA input directory:

```bash
cd code/testCases/EPA_7x11array/QES-files
```

Run:

```bash
../../../../build/qesWinds/qesWinds \
  -q EPA_7x11array_Winds.xml \
  -s 2 \
  -w \
  -t \
  -o ../QES-data/EPA_GPU
```

The principal generated files are:

```text
EPA_GPU_windsOut.nc
EPA_GPU_windsWk.nc
EPA_GPU_turbOut.nc
```

A successful run should report CUDA support and execution of the wall-below GPU calculation. A representative diagnostic is:

```text
GPU work cells with wall below: DONE on GPU
```

### 6.2 QES-Plume GPU smoke test

A short Plume configuration can be used to verify the GPU particle path without running the full EPA simulation.

Run:

```bash
../../../../build/qesPlume/qesPlume \
  -q EPA_7x11array_Plume_smoke.xml \
  -w ../QES-data/EPA_GPU_windsWk.nc \
  -t ../QES-data/EPA_GPU_turbOut.nc \
  -o ../QES-data/EPA_GPU_smoke
```

A successful run should report the selected CUDA device and GPU particle timing diagnostics, for example:

```text
[GPU STAGE1-PERF] Using GPU: ...
[QES-Plume PERF] GPU/CPU timing breakdown
```

The smoke test verifies successful execution of the GPU particle-advection path. It is not a performance benchmark.

### 6.3 Optional convenience script

If `run_gpu_smoke.sh` is retained in the EPA test-case directory, the two stages can be run sequentially from the repository root with:

```bash
./code/testCases/EPA_7x11array/run_gpu_smoke.sh
```

The script checks for the expected QES-Turb and QES-Plume GPU diagnostics and exits with an error if either GPU path is not detected.

## 7. Numerical consistency

### 7.1 QES-Turb

The GPU implementation reformulates the wall-below local mixing-length calculation as a target-cell parallel operation while retaining the original physical calculation.

For the manuscript cases, the local mixing-length results from CPU and GPU implementations are identical. Downstream turbulence variables exhibit only negligible floating-point-level differences.

### 7.2 QES-Plume

QES-Plume is a stochastic Lagrangian particle model.

The CPU and GPU implementations use different random-number generation strategies. Consequently, CPU and GPU particle trajectories represent different stochastic realizations and are not expected to match particle by particle.

CPU-GPU consistency should therefore be evaluated statistically using concentration fields and statistical metrics, as described in the accompanying manuscript.

## 8. Manuscript benchmark configuration and reported performance

The principal high-resolution benchmark described in the manuscript includes:

- horizontal domain: 3 km × 3 km;
- computational grid: 600 × 600 × 96;
- total number of grid cells: 34,560,000;
- main large-particle case: 720,000 released particles;
- primary GPU platform: NVIDIA GeForce RTX 3090.

The principal performance results reported in the manuscript include:

- QES-Turb wall-below calculation: approximately 634 s on CPU and approximately 7.3 s on the RTX 3090 GPU, corresponding to an acceleration factor of about 87;
- QES-Plume particle advection for 720,000 particles: approximately 1045 s on CPU and approximately 21 s on the RTX 3090 GPU, corresponding to an acceleration factor of approximately 50;
- additional RTX 4090 evaluation: the complete reported QES workflow can be completed within approximately one minute.

These timings are hardware dependent and are reported here only as manuscript benchmark results. The public EPA functional test is not intended to reproduce these exact values.

## 9. Typical model outputs

Depending on the selected workflow and configuration, QES generates NetCDF files such as:

```text
windsOut.nc
windsWk.nc
turbOut.nc
plumeOut.nc
```

The exact filename prefix is controlled by the execution command.

## 10. Cleaning generated files

The `build/` directory and generated test-case outputs should not be included in the frozen source archive.

After local verification:

```bash
rm -rf build
rm -f code/testCases/EPA_7x11array/QES-data/*.nc
rm -f code/testCases/EPA_7x11array/QES-data/*.log
```

The EPA input XML files, shapefile geometry, run script, and post-processing utilities should remain.

## 11. License

This archived QES release is distributed under the **GNU General Public License, version 3 (GPL-3.0)**.

See:

```text
LICENSE
```

Original copyright and license notices retained in individual QES source files remain applicable.

## 12. Code and data availability

The exact source version associated with the manuscript should be deposited as a frozen release in a persistent public repository such as Zenodo.

Final software DOI:

```text
[TO BE ADDED AFTER ZENODO ARCHIVING]
```

The DOI cited in the manuscript must refer to the exact frozen software release associated with the submitted model version.

Detailed site-specific information associated with the nuclear facility cannot be made publicly available due to security restrictions. The public archive therefore provides the model source code, GPU implementation, and openly distributable QES test cases.

## 13. Citation

If this software release is used, please cite both the accompanying manuscript and the archived software release.

### Manuscript

**GPU-Accelerated QES-Turb and QES-Plume in QES v2.0.2 for Rapid High-Resolution Atmospheric Dispersion Modeling in Complex Nuclear Power Plant Environments.**

*Geoscientific Model Development*, submitted.

### Software archive

```text
GPU-Accelerated QES-Turb and QES-Plume in QES v2.0.2.
Zenodo. [DOI TO BE ADDED]
```

Complete the final author list and DOI after creation of the frozen Zenodo record.

## 14. Contact

For questions related to the GPU developments and the archived release, please contact the corresponding author listed in the accompanying manuscript.

---

## Final archive checklist

Before uploading the frozen release:

1. Confirm that the source contains no machine-specific hard-coded absolute paths.
2. Confirm that `README.md`, `BUILD_ENVIRONMENT.txt`, `LICENSE`, and `MODIFIED_FILES.txt` are present.
3. Confirm that a clean out-of-source CUDA-enabled build succeeds.
4. Confirm that the EPA QES-Turb test reports execution of the GPU wall-below path.
5. Confirm that the EPA Plume smoke test reports execution of the GPU particle path.
6. Remove generated `build/` files before archiving.
7. Remove generated NetCDF outputs and log files from `code/testCases/EPA_7x11array/QES-data/`.
8. Confirm that no site-specific nuclear-facility input, output, coordinate, path, or metadata files are present.
9. Check the final archive size and file list.
10. Create the frozen Zenodo record.
11. Insert the final Zenodo DOI into this README and into the manuscript's Code and data availability section.
