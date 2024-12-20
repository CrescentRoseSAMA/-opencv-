cmake_minimum_required(VERSION 3.6)
project(start)

set(CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")


find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(CharSeg OBJECT seg.cpp)

target_link_libraries(CharSeg ${OpenCV_LIBS})