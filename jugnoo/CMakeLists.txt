cmake_minimum_required(VERSION 2.6)
project(jugnoo)

find_package( TIFF REQUIRED )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )
include_directories( ${TIFF_INCLUSE_DIR} )

#add_subdirectory(src)

add_executable(jugnoo src/main.cpp)

target_link_libraries( jugnoo  ${TIFF_LIBRARIES} )

