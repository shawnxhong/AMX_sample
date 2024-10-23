cmake_minimum_required(VERSION 3.10)
project(matmul)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O2 -march=native -fno-strict-aliasing")

set(SOURCES matmul.cpp)
add_executable(matmul ${SOURCES})
include_directories(${CMAKE_SOURCE_DIR})
