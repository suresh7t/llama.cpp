set(TARGET llama-hpcache-calibrate)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common moe-hpcache ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# the host arena probe registers memory for the compute GPU directly
if (HPCACHE_WITH_CUDA)
    target_link_libraries(${TARGET} PRIVATE CUDA::cudart_static)
    target_compile_definitions(${TARGET} PRIVATE HPCACHE_CUDA)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
