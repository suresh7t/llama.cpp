set(TARGET moe-hpcache)

add_library(${TARGET} SHARED
    ${PROJECT_SOURCE_DIR}/common/json.cpp
    ${PROJECT_SOURCE_DIR}/common/json.h
    hpcache-io.cpp
    hpcache-io.h
    hpcache-lazy-map.h
    hpcache-placement.cpp
    hpcache-placement.h
    hpcache-scheduler.cpp
    hpcache-scheduler.h
    hpcache.cpp
    hpcache.h
)

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml/src
)
target_link_libraries(${TARGET} PUBLIC ggml vendor::nlohmann vendor::sheredom)
target_compile_features(${TARGET} PUBLIC cxx_std_17)

if (WIN32)
    set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME moe-hpcache)
else()
    set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME moe-hpcache PREFIX "")
endif()

if (LLAMA_BUILD_TOOLS AND LLAMA_BUILD_COMMON)
    add_subdirectory(calibrate)
endif()
