cmake_minimum_required(VERSION 3.20)

project(GPGPUTasks)

set(CMAKE_CXX_STANDARD 17)

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
        libbase
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
        src/kernels/cl/matrix_vector_multiply.cl)

set(OPENCL_INCLUDES src/kernels/defines.h src/kernels/cl/helpers/rassert.cl)
set(OPENCL_DEFINES)
foreach(CL_FILE IN LISTS OPENCL_SOURCES)
    compile_opencl(HEADERS ${CL_FILE} OPENCL_INCLUDES OPENCL_DEFINES)
endforeach()

set(VULKAN_SOURCES
)

set(VULKAN_DEFINES)
foreach(VK_FILE IN LISTS VULKAN_SOURCES)
    compile_vulkan(HEADERS ${VK_FILE} VULKAN_DEFINES)
endforeach()

add_library(${PROJECT_NAME}_core STATIC ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME}_core ${LIBRARIES})

add_executable(main_aplusb                 src/main_aplusb.cpp)
add_executable(main_sparse_matrix_multiply src/main_csr_spmv.cpp)

target_link_libraries(main_aplusb                 ${PROJECT_NAME}_core)
target_link_libraries(main_sparse_matrix_multiply ${PROJECT_NAME}_core)
