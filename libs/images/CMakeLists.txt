cmake_minimum_required(VERSION 3.5)

project(libimages)

# This is thin wrapper around CImg header-only library available at https://github.com/dtschump/CImg (v.2.3.4 0f0d65d984b08ad8178969f4fa4d1641d721354b)

set(HEADERS
        libimages/debug_io.h
        libimages/images.h
        )

set(SOURCES
        libimages/debug_io.cpp
        libimages/images.cpp
        )

set(CMAKE_CXX_STANDARD 17)

set(LIBRARIES libbase)

if(UNIX)
    find_package(X11 REQUIRED)
    set(LIBRARIES ${LIBRARIES} ${X11_LIBRARIES})
endif()

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
