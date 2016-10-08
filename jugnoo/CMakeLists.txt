cmake_minimum_required(VERSION 2.8)
project(jugnoo)

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules 
    )

add_definitions( -std=c++11 -Wall )

set(CMAKE_BUILD_TYPE Release )

#find_package( OPENCV REQUIRED COMPONENTS core highgui imgcodecs )
find_package( TIFF REQUIRED )
find_package( GSL REQUIRED )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_executable(jugnoo main.cpp)

target_link_libraries( jugnoo  
    ${TIFF_LIBRARIES}
    ${GSL_LIBRARIES}
    )
