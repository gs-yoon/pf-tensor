cmake_minimum_required(VERSION 3.8)
project( platform_free_c LANGUAGES C)

include_directories( ${CMAKE_SOURCE_DIR}/includes/
                     ${CMAKE_SOURCE_DIR}/includes/tensor/
                     ${CMAKE_SOURCE_DIR}/includes/definition 
                     ${CMAKE_SOURCE_DIR}/includes/tensor/kernel 
                     ${CMAKE_SOURCE_DIR}/includes/tensor/kernel/t32f 
                     ${CMAKE_SOURCE_DIR}/includes/tensor/kernel/t32f/architecture 
                     )


file(GLOB SOURCES
    "src/*.c"
    "src/tensor/*.c"
    "src/tensor/kernel/*.c"
    "src/tensor/kernel/t32f/*.c"
    "src/tensor/kernel/t32f/architecture/*.c"
    # "src/tensor/*.cpp"
    # "src/tensor/*.cu"
    # "src/essential/basefunc/*.cpp"/
    # "src/essential/network/*.cpp"
    # "src/essential/tensor/*.cpp"
)

add_executable( tensor ${SOURCES} )  #c/cpp 파일 들끼리 link 작업을 하여, executable 파일 생성
add_executable( testo ${PROJECT_SOURCE_DIR}/test.c )  #c/cpp 파일 들끼리 link 작업을 하여, executable 파일 생성
# add_executable( tensorl ${SOURCES} )  #c/cpp 파일 들끼리 link 작업을 하여, executable 파일 생성

# target_link_libraries( tensor ${OpenCV_LIBS} ) # 라이브러리 연결
