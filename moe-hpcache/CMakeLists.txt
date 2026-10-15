set(TARGET moe-hpcache)

set(HPCACHE_SOURCES
    moe-provider.cpp
    ${PROJECT_SOURCE_DIR}/common/json.cpp
    ${PROJECT_SOURCE_DIR}/common/json.h
    moe-provider.cpp
    hpcache-io.cpp
    hpcache-io.h
    hpcache-lazy-map.h
    hpcache-placement.cpp
    hpcache-placement.h
    hpcache-scheduler.cpp
    hpcache-scheduler.h
    hpcache.cpp
    hpcache.h
    hpcache-sync.h
)

# the doorbell decode needs the CUDA runtime for pinned memory and three small kernels
set(HPCACHE_WITH_CUDA OFF)
if (GGML_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "native")
        endif()
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(HPCACHE_WITH_CUDA ON)
        list(APPEND HPCACHE_SOURCES
            hpcache-arena.cpp
            hpcache-arena.h
            hpcache-cuda.cu
            hpcache-cuda.h
            hpcache-cpu.cpp
            hpcache-cpu.h
            hpcache-decode.cpp
            hpcache-decode.h
        )
    endif()
endif()

add_library(${TARGET} SHARED ${HPCACHE_SOURCES})

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml/src
)
target_link_libraries(${TARGET} PUBLIC ggml vendor::nlohmann vendor::sheredom)
target_compile_features(${TARGET} PUBLIC cxx_std_17)

if (HPCACHE_WITH_CUDA)
    # the shared runtime, like ggml-cuda: one runtime instance and one context per process
    target_link_libraries(${TARGET} PUBLIC CUDA::cudart)
    target_compile_definitions(${TARGET} PRIVATE HPCACHE_CUDA)
endif()

if (WIN32)
    # WaitOnAddress and WakeByAddress (hpcache-sync.h)
    target_link_libraries(${TARGET} PRIVATE synchronization)
    set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME moe-hpcache)
else()
    set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME moe-hpcache PREFIX "")
endif()

if (LLAMA_BUILD_TOOLS AND LLAMA_BUILD_COMMON)
    add_subdirectory(calibrate)
endif()
