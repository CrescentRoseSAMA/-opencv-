cmake_minimum_required(VERSION 3.6)
project(start)

set(CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE AUTOUIC ON)

add_subdirectory(charseg)
add_subdirectory(Infer)
add_subdirectory(MvCamera)
find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Widgets REQUIRED)

include_directories(${INCLUDEPATH})
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(main qui.cpp main.cpp)

target_link_libraries(main Qt5::Core Qt5::Widgets ${OpenCV_LIBS} TRTFrame CharSeg MvCamera)