cmake_minimum_required(VERSION 3.5)

project(GPGPUTasks)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(libs)

# Компилируем исполняемый файл нашей программы
add_executable(enumDevices src/main.cpp)

# Mac is not supporting OpenCL oficially, but actually has backwards compatability
target_compile_definitions(enumDevices PRIVATE CL_TARGET_OPENCL_VERSION=120)

# Мы линкуемся с библиотекой clew (см. папку libs/clew) - она содержит в себе OpenCL headers + позволяет в runtime слинковаться с OpenCL API функциями
target_link_libraries(enumDevices libclew)
