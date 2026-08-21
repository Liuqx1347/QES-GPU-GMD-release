include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#INCLUDE_DIRECTORIES (${CMAKE_SOURCE_DIR}/quicutil)  
#LINK_DIRECTORIES (${CMAKE_BINARY_DIR}/quicutil/)

SET(qesPlumeCoreSources
    Random.cpp
    RandomSingleton.cpp
    handlePlumeArgs.cpp

    Interp.cpp
    InterpNearestCell.cpp
    InterpPowerLaw.cpp
    InterpTriLinear.cpp

    Deposition.cpp

    Plume.cpp
    AdvectParticle.cpp
    DepositParticle.cpp

    PlumeOutput.cpp
    PlumeOutputParticleData.cpp

    BoundaryConditions.hpp
    Particle.hpp
    ParseParticle.hpp
    ParticleFactories.hpp
    ParticleSmall.hpp
    ParticleLarge.hpp
    ParticleHeavyGas.hpp

    ReleaseType.hpp
    ReleaseType_instantaneous.hpp
    ReleaseType_continuous.hpp
    ReleaseType_duration.hpp

    Sources.hpp
    SourceType.hpp
    SourcePoint.cpp SourcePoint.hpp
    SourceLine.cpp SourceLine.hpp
    SourceCircle.cpp SourceCircle.hpp
    SourceCube.cpp SourceCube.hpp
    SourceFullDomain.cpp SourceFullDomain.hpp

    DomainBoundaryConditions.cpp

    WallReflection.cpp
    WallReflection_StairStep.cpp
    WallReflection_TriMesh.cpp
)

if (HAS_CUDA_SUPPORT)
  list(APPEND qesPlumeCoreSources AdvectParticleGPU_demo.cu)

  find_package(CUDA REQUIRED)          # 你工程本来就在用 FindCUDA
  include_directories(${CUDA_INCLUDE_DIRS})

  CUDA_ADD_LIBRARY(qesplumecore ${qesPlumeCoreSources})
  target_compile_definitions(qesplumecore PUBLIC HAS_CUDA_SUPPORT=1)
  target_link_libraries(qesplumecore PUBLIC ${CUDA_LIBRARIES})
else()
  add_library(qesplumecore ${qesPlumeCoreSources})
endif()

link_directories(/usr/local/lib /lib/x86_64-linux-gnu)
target_link_libraries(qesplumecore PUBLIC png)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
  target_link_libraries(qesplumecore PUBLIC OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP NOT found. Building without OpenMP.")
endif()


# Uncomment below if additional libraries are required
# target_link_libraries(qesplumecore PUBLIC quicloader)
# target_link_libraries(qesplumecore PUBLIC util)
# target_link_libraries(qesplumecore PUBLIC ${Boost_PROGRAM_OPTIONS_LIBRARIES})

#if (APPLE)
#  find_package(OpenGL)
#  INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR})
#  find_package(GLUT)
#  INCLUDE_DIRECTORIES(${GLUT_INCLUDE_DIR})
#  target_link_libraries (qesplumecore PUBLIC ${OPENGL_LIBRARIES})
#  target_link_libraries (qesplumecore PUBLIC ${GLUT_LIBRARIES})
#else (APPLE)
#  target_link_libraries (qesplumecore PUBLIC GL)
#  target_link_libraries (qesplumecore PUBLIC glut)
#  target_link_libraries (qesplumecore PUBLIC GLU)
#endif (APPLE)

# target_link_libraries (qesplumecore PUBLIC GLEW)
