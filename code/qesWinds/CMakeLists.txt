# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

IF ($CACHE{HAS_CUDA_SUPPORT})
    CUDA_ADD_EXECUTABLE(qesWinds
        qesWindsMain.cpp
    )
ELSE ($CACHE{HAS_CUDA_SUPPORT})
    ADD_EXECUTABLE(qesWinds
        qesWindsMain.cpp
    )
ENDIF ($CACHE{HAS_CUDA_SUPPORT})



# 链接 qeswindscore 和其他依赖
target_link_libraries(qesWinds qeswindscore)
IF ($CACHE{HAS_CUDA_SUPPORT})
    target_link_libraries(qesWinds qeswindsgpu)
ENDIF()
target_link_libraries(qesWinds qesutil)

IF ($CACHE{HAS_OPTIX_SUPPORT})
    target_link_libraries(qesWinds qesOptix)
ENDIF()

# 链接外部库
link_external_libraries(qesWinds)

find_package(PNG REQUIRED)

target_link_libraries(qesWinds  ${PNG_LIBRARIES})

