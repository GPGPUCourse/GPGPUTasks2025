cmake_minimum_required(VERSION 3.5)

project(libclew)

set(HEADERS
        CL/cl.h
        CL/cl_d3d10.h
        CL/cl_ext.h
        CL/cl_gl.h
        CL/cl_gl_ext.h
        CL/cl_platform.h
        CL/cl_version.h
        CL/opencl.h
        libclew/ocl_init.h
        )

set(SOURCES
        libclew/ocl_init.cpp
        )

add_library(${PROJECT_NAME} ${HEADERS} ${SOURCES})
target_link_libraries (${PROJECT_NAME} ${CMAKE_DL_LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
