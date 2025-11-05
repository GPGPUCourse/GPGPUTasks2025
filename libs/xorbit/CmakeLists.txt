cmake_minimum_required(VERSION 3.5)

project(libxorbit)

set(HEADERS
        libxorbit/contexted_timer.h
)

set(SOURCES
        libxorbit/contexted_timer.cpp
)

set(CMAKE_CXX_STANDARD 17)

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} Threads::Threads libutils)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})