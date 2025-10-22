cmake_minimum_required(VERSION 3.20)

project(GPGPUTasks)

set(CMAKE_CXX_STANDARD 17)

# Если вы установили CUDA SDK - можете включить поддержку CUDA
option(GPU_CUDA_SUPPORT "CUDA support." OFF)

# GLSLC_BIN используйтся для компиляции Vulkan шейдеров (GLSL)
if (NOT DEFINED GLSLC_BIN)
    # В случае Linux глобальный glslc будет использован (вы его установили среди прочего в scripts/linux/install_vulkan_sdk.sh)
    set(GLSLC_BIN glslc)
    message(STATUS "Vulkan glslc compiler from system PATH will be used")
else ()
    # В случае Windows вам нужно указать его через CMake options: -DSPIR_CLANG_BIN="C:\Program Files\LLVM\bin\clang.exe" (установлен по инструкции scripts/windows/install_dependencies.txt)
    message(STATUS "Vulkan glslc compiler is specified")
endif ()

# GTest позволяет писать удобные unit-test-ы
find_package(GTest REQUIRED)
set(GTEST_LIBRARIES GTest::gtest)
set(GTEST_MAIN_LIBRARIES GTest::gtest_main)
set(GTEST_BOTH_LIBRARIES ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

# OpenMP позволит распараллеливать циклы на все ядра процессора простыми директивами
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    # Это не ошибка т.к. если нет OpenMP - код будет работать, просто директивы распараллеливания ничего не будут делать
    message(WARNING "OpenMP not found!")
endif ()

add_subdirectory(libs)

set(HEADERS
        src/kernels/kernels.h
)

set(SOURCES
        src/kernels/kernels.cpp
)

set(LIBRARIES
        libclew
        libgpu
        libimages
        libutils
        libjson
)

# compile_opencl CMake-функция объявлена в libs/gpu/libgpu/CMakeLists.txt:157
# Она считывает все байты из файла src/kernels/cl/aplusb.cl (т.е. весь исходный код кернела), выполняет стадию препроцессора с помощью OPENCL_CXX_COMPILER компилятора (clang или gcc) - в т.ч. выполняя все #include
# Получившийся после препроцессора исходник находится в файле src/kernels/cl/generated_kernels/aplusb_nospir_opencl120.cl
# Затем получивщийся исходник преобразуется в массив байтов в файле src/kernels/cl/generated_kernels/aplusb_nospir_opencl120.h
# (чтобы не нужно было в runtime читать исходник с диска, чтобы был монолитный исполняемый файл)
# Обратите внимание что это происходит на этапе компиляции, кроме того эта команда делает так чтобы файл src/kernels/vk/aplusb.h был перечислен среди HEADERS для компиляции при вызове add_executable
set(OPENCL_SOURCES
        src/kernels/cl/aplusb.cl
        src/kernels/cl/prefix_sum_01_reduction.cl
        src/kernels/cl/prefix_sum_02_prefix_accumulation.cl)

set(OPENCL_INCLUDES src/kernels/defines.h src/kernels/cl/helpers/rassert.cl)
set(OPENCL_DEFINES)
foreach(CL_FILE IN LISTS OPENCL_SOURCES)
    compile_opencl(HEADERS ${CL_FILE} OPENCL_INCLUDES OPENCL_DEFINES)
endforeach()

# compile_vulkan CMake-функция объявлена в libs/gpu/libgpu/vulkan/CMakeLists.txt:97
# Она считывает все байты из файла src/kernels/vk/aplusb.comp (т.е. весь исходный код кернела), компилирует с помощью glslc компилятора (в т.ч. выполняя все #include)
# Скомпилированный SPIR байткод находится в файле src/kernels/vk/generated_kernels/aplusb_comp_spirv_vulkan.spir
# SPIR байткод можно проанализировать через рефлексию - запустив /usr/local/bin/spirv-reflect src/vk/generated_kernels/aplusb_comp_spirv_vulkan.spir
# Затем получившийся скомпилированный SPIR байткод преобразует в массив байтов в файле src/kernels/vk/generated_kernels/aplusb_comp_spirv_vulkan.h
# (чтобы не нужно было в runtime читать шейдер/байткод с диска, чтобы был монолитный исполняемый файл)
# Обратите внимание что это происходит на этапе компиляции, кроме того эта команда делает так чтобы файл src/kernels/vk/aplusb_comp.h был перечислен среди HEADERS для компиляции при вызове add_executable
set(VULKAN_SOURCES
        src/kernels/vk/aplusb.comp
        src/kernels/vk/fill_buffer_with_zeros.comp
        src/kernels/vk/prefix_sum_01_reduction.comp
        src/kernels/vk/prefix_sum_02_prefix_accumulation.comp)

set(VULKAN_DEFINES)
foreach(VK_FILE IN LISTS VULKAN_SOURCES)
    compile_vulkan(HEADERS ${VK_FILE} VULKAN_DEFINES)
endforeach()

if (GPU_CUDA_SUPPORT)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Компиляция CUDA происходит NVCC компилятором, все файлы с расширением .cu компилируются два раза:
    # 1) один раз обычным компилятором - для HOST-side кода, компилируются только те функции которые не помечены специально что они для видеокарты (или помечены что они и для видеокарты и для HOST-side кода)
    # 2) второй раз специальным CUDA-компилятором NVCC - для DEVICE-side кода, например функции помеченные специальным модификатором __global__ (кернелы) или __device__ (функции которые можно вызвать на видеокарте, т.е. из кернелов)
    # Вероятно вам стоит выставить в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo (первое - чтобы включить поддержку WMMA, второе - чтобы compute-sanitizer и профилировщик знали номера строк кернела)
    set(CUDA_SOURCES
            src/kernels/cu/aplusb.cu
            src/kernels/cu/fill_buffer_with_zeros.cu
            src/kernels/cu/prefix_sum_01_reduction.cu
            src/kernels/cu/prefix_sum_02_prefix_accumulation.cu)

    add_definitions(-DCUDA_SUPPORT)
    add_library(${PROJECT_NAME}_core STATIC ${SOURCES} ${CUDA_SOURCES} ${HEADERS})
    target_link_libraries(${PROJECT_NAME}_core ${LIBRARIES})
    target_include_directories(${PROJECT_NAME}_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
else ()
    add_library(${PROJECT_NAME}_core STATIC ${SOURCES} ${HEADERS})
    target_link_libraries(${PROJECT_NAME}_core ${LIBRARIES})
endif ()

add_executable(main_aplusb     src/main_aplusb.cpp)
add_executable(main_prefix_sum src/main_prefix_sum.cpp)

target_link_libraries(main_aplusb     ${PROJECT_NAME}_core)
target_link_libraries(main_prefix_sum ${PROJECT_NAME}_core)

if (GPU_CUDA_SUPPORT)
    target_include_directories(main_aplusb     PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_include_directories(main_prefix_sum PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif ()
