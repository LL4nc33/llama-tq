include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-tools.cpp
    server-tools.h
    server-anthropic-cache.cpp
    server-anthropic-cache.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC common mtmd ${CMAKE_THREAD_LIBS_INIT})


# llama-server executable

set(TARGET llama-server)

set(TARGET_SRCS
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)

option(LLAMA_BUILD_WEBUI "Build the embedded Web UI" ON)

if (LLAMA_BUILD_WEBUI)
    set(PUBLIC_ASSETS
        index.html
        bundle.js
        bundle.css
        loading.html
    )

    foreach(asset ${PUBLIC_ASSETS})
        set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
        set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
        list(APPEND TARGET_SRCS ${output})
        add_custom_command(
            DEPENDS "${input}"
            OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
        )
        set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
    endforeach()
    add_definitions(-DLLAMA_BUILD_WEBUI)
else()
endif()

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

# Opt-in gzip compression for non-streaming HTTP responses (LLAMA_SERVER_ZLIB=ON).
# Streaming SSE chunks are not compressed — per-chunk gzip overhead exceeds the
# savings on tiny token deltas, and some SSE clients choke on gzipped streams.
# For streaming, prefer ssh -C / transport-level compression.
option(LLAMA_SERVER_ZLIB "Enable gzip/deflate for non-streaming HTTP responses" ON)
if (LLAMA_SERVER_ZLIB)
    find_package(ZLIB QUIET)
    if (ZLIB_FOUND)
        target_compile_definitions(${TARGET} PRIVATE CPPHTTPLIB_ZLIB_SUPPORT)
        target_link_libraries(${TARGET} PRIVATE ZLIB::ZLIB)
        message(STATUS "llama-server: gzip response compression enabled (zlib ${ZLIB_VERSION_STRING})")
    else()
        message(STATUS "llama-server: zlib not found, gzip compression disabled")
    endif()
endif()

target_compile_features(${TARGET} PRIVATE cxx_std_17)
