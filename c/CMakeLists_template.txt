cmake_minimum_required(VERSION 3.8)
project( handmade_learning LANGUAGES CXX)
find_package( OpenCV 4.0 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories( ${CMAKE_SOURCE_DIR}/dependencies/eigen-3.4.0
                        ${CMAKE_SOURCE_DIR}/src/essential/basis/
                        ${CMAKE_SOURCE_DIR}/src/essential/tensor/
                        ${CMAKE_SOURCE_DIR}/src/essential/model/
                        #${CMAKE_SOURCE_DIR}/src/essential/cuda_utils/
                        ${CMAKE_SOURCE_DIR}/src/includes/definition )


file(GLOB SOURCES
    "src/*.cpp"
    "src/*.cu"
    "src/essential/basefunc/*.cpp"
    "src/essential/network/*.cpp"
    "src/essential/tensor/*.cpp"
)
add_executable( handmade_learning ${SOURCES} )

target_link_libraries( handmade_learning ${OpenCV_LIBS} )

