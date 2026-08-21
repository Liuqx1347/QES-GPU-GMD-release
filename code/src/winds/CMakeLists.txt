include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

SET( qesWindCoreSources
  Canopy.cpp
  CanopyElement.cpp
  CanopyHomogeneous.cpp
  CanopyIsolatedTree.cpp
  CanopyWindbreak.cpp
  CanopyROC.cpp
  Cell.cpp
  CPUSolver.cpp
  Cut_cell.cpp
  DTEHeightField.cpp
  handleWINDSArgs.cpp
  LocalMixing.cpp
  LocalMixingDefault.cpp
  LocalMixingNetCDF.cpp
  LocalMixingSerial.cpp
  LocalMixingOptix.cpp
  PolyBuilding.cpp PolyBuilding.h
  Sensor.cpp
  Solver.cpp
  Solver_CPU_RB.cpp
  TURBParams.h
  TURBGeneralData.cpp
  TURBGeneralData.h
  TURBOutput.cpp
  TURBWall.cpp
  TURBWallBuilding.cpp
  TURBWallTerrain.cpp
  WindProfilerSensorType.cpp
  WindProfilerBarnCPU.cpp
  WindProfilerBarnGPU.cpp
  WindProfilerWRF.cpp
  WINDSInputData.h
  WINDSGeneralData.cpp WINDSGeneralData.h
  WINDSOutputVisualization.cpp
  WINDSOutputWorkspace.cpp
  WRFInput.cpp WRFInput.h
  Wall.cpp Wall.h
  UpwindCavity.cpp
  NonLocalMixing.cpp
  PolygonWake.cpp
  StreetCanyon.cpp
  StreetCanyonModified.cpp
  Sidewall.cpp
  Rooftop.cpp
  StreetIntersection.cpp
  TimeSeries.h
  )

IF ($CACHE{HAS_CUDA_SUPPORT})
   CUDA_ADD_LIBRARY( qeswindscore
      ${qesWindCoreSources}
   )
   target_compile_definitions(qeswindscore PUBLIC HAS_CUDA_SUPPORT=1)
ELSE ($CACHE{HAS_CUDA_SUPPORT})
   ADD_LIBRARY( qeswindscore
      ${qesWindCoreSources}
   )
ENDIF ($CACHE{HAS_CUDA_SUPPORT})


IF ($CACHE{HAS_CUDA_SUPPORT})
  CUDA_ADD_LIBRARY(qeswindsgpu
    DynamicParallelism.cu
    GlobalMemory.cu GlobalMemory.h
    SharedMemory.cu SharedMemory.h
    TURBGeneralData.cu
    WindProfilerBarnGPU.cu
    LocalMixingSerialGPU.cu LocalMixingSerialGPU.h)

  target_compile_definitions(qeswindsgpu PUBLIC HAS_CUDA_SUPPORT=1)
ENDIF ($CACHE{HAS_CUDA_SUPPORT})


