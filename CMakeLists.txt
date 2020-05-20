cmake_minimum_required(VERSION 2.8)
project(LK_optical_flow)

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
add_definitions(-std=c++11 -O3 -g -DNDEBUG)

find_package( OpenCV REQUIRED )

include_directories(
  include
${OpenCV_INCLUDE_DIRS}
)
add_library(model_detector
src/model_detector.cpp)
target_link_libraries(model_detector
${OpenCV_LIBS}
)

add_executable(model_detect_test src/main.cpp)
target_link_libraries(model_detect_test
model_detector
${OpenCV_LIBS}
)

