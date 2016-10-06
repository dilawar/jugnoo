cmake_minimum_required(VERSION 2.8)
project(jugnoo)

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules 
    )

find_package( OPENCV REQUIRED COMPONENTS core highgui )
find_package( TIFF REQUIRED )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

#add_subdirectory(src)

add_executable(jugnoo src/main.cpp)

target_link_libraries( jugnoo  
    ${TIFF_LIBRARIES}
    ${OPENCV_LIBRARIES} 
    )
