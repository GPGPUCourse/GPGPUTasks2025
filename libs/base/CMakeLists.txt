cmake_minimum_required(VERSION 3.5)

project(libbase)

set(HEADERS
        libbase/data_type.h
        libbase/fast_random.h
        libbase/gtest_utils.h
        libbase/math.h
        libbase/omp_utils.h
        libbase/platform.h
        libbase/point.h
        libbase/runtime_assert.h
        libbase/string_utils.h
        libbase/thread_mutex.h
        libbase/timer.h
        )

set(SOURCES
        libbase/gtest_utils.cpp
        libbase/omp_utils.cpp
        libbase/string_utils.cpp
        libbase/thread_mutex.cpp
        )

set(CMAKE_CXX_STANDARD 17)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} Threads::Threads ${GTEST_BOTH_LIBRARIES} ${CMAKE_DL_LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
