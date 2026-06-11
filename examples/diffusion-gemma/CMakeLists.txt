set(TARGET llama-diffusion-gemma-cli)
add_executable(${TARGET} diffusion-gemma-cli.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama common mtmd ${CMAKE_THREAD_LIBS_INIT})
# mtmd (tools/) is added after examples/, so add its include dir explicitly for the headers
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/mtmd
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/imatrix)   # imatrix-collector.h (decoder-path imatrix)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# OpenAI-compatible HTTP server for the block-diffusion models (llama-server analogue)
set(TARGET llama-diffusion-gemma-server)

# Embed the same WebUI as llama-server so the diffusion server serves the chat UI on "/".
# The assets live in tools/server/public; generate the .hpp blobs into this binary dir via the
# shared xxd.cmake, mirroring tools/server/CMakeLists.txt.
set(DG_SERVER_SRCS diffusion-gemma-server.cpp)
option(LLAMA_BUILD_WEBUI "Build the embedded Web UI" ON)
if (LLAMA_BUILD_WEBUI)
    set(DG_PUBLIC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/server/public")
    set(DG_PUBLIC_ASSETS index.html bundle.js bundle.css loading.html)
    foreach(asset ${DG_PUBLIC_ASSETS})
        set(input  "${DG_PUBLIC_DIR}/${asset}")
        set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
        list(APPEND DG_SERVER_SRCS ${output})
        add_custom_command(
            DEPENDS "${input}"
            OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake")
        set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
    endforeach()
endif()

add_executable(${TARGET} ${DG_SERVER_SRCS})
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama common mtmd cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/mtmd
    ${CMAKE_CURRENT_BINARY_DIR}    # generated index.html.hpp etc.
    ${CMAKE_SOURCE_DIR}/vendor)   # cpp-httplib/httplib.h, nlohmann/json.hpp
if (LLAMA_BUILD_WEBUI)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_BUILD_WEBUI)
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_17)
