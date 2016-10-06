cmake_minimum_required(VERSION 2.6)
project(jugnoo)

#find_package()

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_subdirectory(src)

add_executable(jugnoo main.cpp)

target_link_libraries( jugnoo )

