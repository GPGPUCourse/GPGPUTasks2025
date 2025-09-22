cmake_minimum_required(VERSION 3.5)

project(GPGPUTasks)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(libs)

# Компилируем исполняемый файл нашей программы
add_executable(enumDevices src/main.cpp)

if(APPLE)
    target_link_libraries(enumDevices "-framework OpenCL")
else()
    target_link_libraries(enumDevices libclew)
endif()
