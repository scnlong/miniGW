cmake_minimum_required(VERSION 3.20)
project(gw_prototype LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GW_ENABLE_NATIVE "Enable -march=native when supported" OFF)

add_library(gwcore
    src/frequency_grids.cpp
    src/pade.cpp
    src/npy.cpp
    src/gw.cpp
    src/io.cpp
)

target_include_directories(gwcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(gwcore PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
)

if(GW_ENABLE_NATIVE)
    target_compile_options(gwcore PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native>)
endif()

add_executable(gw src/main.cpp)
target_link_libraries(gw PRIVATE gwcore)
