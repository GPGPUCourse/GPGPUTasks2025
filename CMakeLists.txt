cmake_minimum_required(VERSION 3.5)

project(GPGPUTasks)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(libs)

# Компилируем исполняемый файл нашей программы
add_executable(enumDevices src/main.cpp
        src/main.cpp)

# Мы линкуемся с библиотекой clew (см. папку libs/clew) - она содержит в себе OpenCL headers + позволяет в runtime слинковаться с OpenCL API функциями
target_link_libraries(enumDevices libclew)
