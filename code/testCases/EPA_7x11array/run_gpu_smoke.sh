#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BUILD_DIR="${QES_BUILD_DIR:-${REPO_ROOT}/build}"

QES_FILES="${SCRIPT_DIR}/QES-files"
QES_DATA="${SCRIPT_DIR}/QES-data"

WINDS_EXE="${BUILD_DIR}/qesWinds/qesWinds"
PLUME_EXE="${BUILD_DIR}/qesPlume/qesPlume"

WINDS_XML="EPA_7x11array_Winds.xml"
PLUME_XML="EPA_7x11array_Plume_smoke.xml"

mkdir -p "${QES_DATA}"

if [[ ! -x "${WINDS_EXE}" ]]; then
  echo "ERROR: qesWinds executable not found: ${WINDS_EXE}" >&2
  exit 1
fi

if [[ ! -x "${PLUME_EXE}" ]]; then
  echo "ERROR: qesPlume executable not found: ${PLUME_EXE}" >&2
  exit 1
fi

if [[ ! -f "${QES_FILES}/${WINDS_XML}" ]]; then
  echo "ERROR: missing ${QES_FILES}/${WINDS_XML}" >&2
  exit 1
fi

if [[ ! -f "${QES_FILES}/${PLUME_XML}" ]]; then
  echo "ERROR: missing ${QES_FILES}/${PLUME_XML}" >&2
  exit 1
fi

rm -f "${QES_DATA}"/EPA_GPU_*.nc "${QES_DATA}"/EPA_GPU_*.log

echo "=== EPA GPU test: QES-Winds + QES-Turb ==="
(
  cd "${QES_FILES}"
  "${WINDS_EXE}" \
    -q "${WINDS_XML}" \
    -s 2 \
    -w \
    -t \
    -o ../QES-data/EPA_GPU \
    2>&1 | tee ../QES-data/EPA_GPU_winds_turb.log
)

if ! grep -q "GPU work cells with wall below: DONE on GPU" \
  "${QES_DATA}/EPA_GPU_winds_turb.log"; then
  echo "ERROR: QES-Turb GPU wall-below path was not detected." >&2
  exit 2
fi

echo "QES-Turb GPU wall-below path: PASS"

echo "=== EPA GPU test: QES-Plume smoke test ==="
(
  cd "${QES_FILES}"
  "${PLUME_EXE}" \
    -q "${PLUME_XML}" \
    -w ../QES-data/EPA_GPU_windsWk.nc \
    -t ../QES-data/EPA_GPU_turbOut.nc \
    -o ../QES-data/EPA_GPU_smoke \
    2>&1 | tee ../QES-data/EPA_GPU_plume_smoke.log
)

if ! grep -q "\[GPU STAGE1-PERF\] Using GPU:" \
  "${QES_DATA}/EPA_GPU_plume_smoke.log"; then
  echo "ERROR: QES-Plume GPU particle path was not detected." >&2
  exit 3
fi

echo "QES-Plume GPU particle path: PASS"
echo "=== EPA GPU functional test completed successfully ==="
