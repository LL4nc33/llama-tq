cmake_minimum_required(VERSION 3.14)
project(trellis_phase1 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(trellis_phase1
    trellis_main.c
    trellis_common.c
    trellis_stubs.c
)

if(NOT WIN32)
    target_link_libraries(trellis_phase1 PRIVATE m)
endif()

target_compile_options(trellis_phase1 PRIVATE -O3 -Wall -Wextra)
