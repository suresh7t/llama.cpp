set(TARGET llama-hpcache-calibrate)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common moe-hpcache ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
