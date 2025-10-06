cmake_minimum_required(VERSION 3.20)

project(libgpu)

set(HEADERS
        libgpu/opencl/device_info.h
        libgpu/opencl/engine.h
        libgpu/opencl/enum.h
        libgpu/opencl/exceptions.h
        libgpu/opencl/utils.h
        libgpu/vulkan/spirv_reflect/include/spirv/unified1/spirv.h # https://github.com/KhronosGroup/SPIRV-Reflect/blob/vulkan-sdk-1.3.280.0/include/spirv/unified1/spirv.h
        libgpu/vulkan/spirv_reflect/shader_module_info.h
        libgpu/vulkan/spirv_reflect/spirv_reflect.h # https://github.com/KhronosGroup/SPIRV-Reflect/blob/vulkan-sdk-1.3.280.0/spirv_reflect.h
        libgpu/vulkan/tests/kernels/kernels.h
        libgpu/vulkan/tests/test_utils.h
        libgpu/vulkan/data_buffer.h
        libgpu/vulkan/data_image.h
        libgpu/vulkan/device.h
        libgpu/vulkan/engine.h
        libgpu/vulkan/enum.h
        libgpu/vulkan/utils.h
        libgpu/vulkan/vulkan_api_headers.h
        libgpu/context.h
        libgpu/device.h
        libgpu/device_memory_pool.h
        libgpu/gold_helpers.h
        libgpu/shared_device_buffer.h
        libgpu/shared_device_image.h
        libgpu/shared_host_buffer.h
        libgpu/utils.h
        libgpu/work_size.h
        )

set(SOURCES
        libgpu/opencl/device_info.cpp
        libgpu/opencl/engine.cpp
        libgpu/opencl/enum.cpp
        libgpu/opencl/utils.cpp
        libgpu/vulkan/spirv_reflect/shader_module_info.cpp
        libgpu/vulkan/spirv_reflect/spirv_reflect.cpp # https://github.com/KhronosGroup/SPIRV-Reflect/blob/vulkan-sdk-1.3.280.0/spirv_reflect.c
        libgpu/vulkan/vk/common_host.cpp
        libgpu/vulkan/data_buffer.cpp
        libgpu/vulkan/data_image.cpp
        libgpu/vulkan/device.cpp
        libgpu/vulkan/engine.cpp
        libgpu/vulkan/enum.cpp
        libgpu/vulkan/utils.cpp
        libgpu/vulkan/vulkan_api_headers.cpp
        libgpu/context.cpp
        libgpu/device.cpp
        libgpu/device_memory_pool.cpp
        libgpu/gold_helpers.cpp
        libgpu/shared_device_buffer.cpp
        libgpu/shared_device_image.cpp
        libgpu/shared_host_buffer.cpp
        libgpu/utils.cpp
        )

set(CUDA_HEADERS
        libgpu/cuda/sdk/helper_math.h
        libgpu/cuda/cuda_api.h
        libgpu/cuda/enum.h
        libgpu/cuda/exceptions.h
        libgpu/cuda/utils.h
        )

set(CUDA_SOURCES
        libgpu/cuda/cuda_api.cpp
        libgpu/cuda/enum.cpp
        libgpu/cuda/utils.cpp
        )

set (SOURCES_TEST
        libgpu/opencl/tests/kernels/kernels.cpp
        libgpu/opencl/tests/aplusb_test.cpp
        libgpu/vulkan/tests/kernels/kernels.cpp
        libgpu/vulkan/tests/aplusb_test.cpp
        libgpu/vulkan/tests/atomic_add_test.cpp
        libgpu/vulkan/tests/batched_binary_search_test.cpp
        libgpu/vulkan/tests/buffers_magic_guards_test.cpp
        libgpu/vulkan/tests/image_conversion_from_float_to_T_test.cpp
        libgpu/vulkan/tests/interpolation_test.cpp
        libgpu/vulkan/tests/main.cpp
        libgpu/vulkan/tests/rasterization_blending_test.cpp
        libgpu/vulkan/tests/rasterization_test.cpp
        libgpu/device_test.cpp
        )

option(GPU_CUDA_SUPPORT "CUDA support." OFF)

set (LIBRARIES
        libbase
        libimages
        libclew
        libutils)

set(CMAKE_CXX_STANDARD 17)

add_executable(hexdumparray libgpu/hexdumparray.cpp)

function(convertIntoHeader sourceFile headerFile arrayName)
    add_custom_command(
            OUTPUT ${PROJECT_SOURCE_DIR}/${headerFile}

            COMMAND ${CMAKE_COMMAND} -E remove ${PROJECT_SOURCE_DIR}/${headerFile}
            COMMAND hexdumparray ${PROJECT_SOURCE_DIR}/${sourceFile} ${PROJECT_SOURCE_DIR}/${headerFile} ${arrayName}

            DEPENDS ${PROJECT_SOURCE_DIR}/${sourceFile} hexdumparray
    )
endfunction()

function (compile_opencl_nospir newoutfiles program includefiles defines ocl_major ocl_minor)
    get_filename_component (CL_NAME ${program} NAME_WE)
    get_filename_component (CL_DIR ${program} DIRECTORY)
    set (CL_DIR "${CL_DIR}/generated_kernels")
    if (NOT "${${defines}}" STREQUAL "")
        join_defines(${defines} defines_str)
    endif()

    set (SPIR_BINARY ${PROJECT_SOURCE_DIR}/${CL_DIR}/${CL_NAME}${defines_str}_nospir_opencl${ocl_major}${ocl_minor}0.cl)
    set (CL_HEADER ${PROJECT_SOURCE_DIR}/${CL_DIR}/${CL_NAME}${defines_str}_nospir_opencl${ocl_major}${ocl_minor}0.h)

    set (VARIABLE_SUFFIX ${CL_NAME}${defines_str}_nospir_opencl${ocl_major}${ocl_minor}0)

    if (MSVC)
        set (OPENCL_CXX_COMPILER ${SPIR_CLANG_BIN})
    else ()
        set (OPENCL_CXX_COMPILER ${CMAKE_CXX_COMPILER})
    endif ()

    add_custom_command (
            OUTPUT  ${CL_HEADER}

            COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/${CL_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/${CL_DIR}

            COMMAND ${OPENCL_CXX_COMPILER} -x c++ -E ${${defines}} -D__OPENCL_VERSION__=${ocl_major}${ocl_minor}0 -DSTATIC_KEYWORD=static ${PROJECT_SOURCE_DIR}/${program} -I${PROJECT_SOURCE_DIR} -o ${SPIR_BINARY}

            COMMAND ${CMAKE_COMMAND} -E remove ${CL_HEADER}

            COMMAND ${CMAKE_COMMAND} -E echo \#include<libgpu/opencl/engine.h> > ${CL_HEADER}
            COMMAND ${CMAKE_COMMAND} -E echo namespace ocl { >> ${CL_HEADER}

            COMMAND hexdumparray ${SPIR_BINARY} ${CL_HEADER} opencl_${VARIABLE_SUFFIX}

            COMMAND ${CMAKE_COMMAND} -E echo static const ocl::VersionedBinary opencl_binary_${VARIABLE_SUFFIX}\(opencl_${VARIABLE_SUFFIX}, sizeof\(opencl_${VARIABLE_SUFFIX}\), 0, ${ocl_major}, ${ocl_minor}\)\; >> ${CL_HEADER}

            COMMAND ${CMAKE_COMMAND} -E echo } >> ${CL_HEADER}

            DEPENDS hexdumparray ${program} ${${includefiles}}
            VERBATIM
    )

    set (${newoutfiles} ${${newoutfiles}} ${CL_HEADER} PARENT_SCOPE)
endfunction (compile_opencl_nospir)

function (compile_opencl outfiles program includefiles defines)
    compile_opencl_nospir(tmp ${program} ${includefiles} ${defines} 1 2)

    set(${outfiles} ${${outfiles}} ${tmp} PARENT_SCOPE)

    get_filename_component (CL_NAME ${program} NAME_WE)
    get_filename_component (CL_DIR ${program} DIRECTORY)
    set (CL_DIR "${CL_DIR}/generated_kernels")
    if (NOT "${${defines}}" STREQUAL "")
        join_defines(${defines} defines_str)
    endif()

    set (PROGRAM_NAME ${CL_NAME}${defines_str})
    set (CL_HEADER ${PROJECT_SOURCE_DIR}/${CL_DIR}/${PROGRAM_NAME}.h)
    configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/libgpu/opencl/cl/opencl_program.h.in ${CL_HEADER})
endfunction (compile_opencl)

set (OPENCL_SOURCES)
compile_opencl(HEADERS libgpu/opencl/cl/dummy_kernel.cl OPENCL_SOURCES DEFINES)
compile_opencl(HEADERS libgpu/opencl/tests/kernels/aplusb.cl OPENCL_SOURCES DEFINES)

set(IMAGE_NCHANNELS 1 2 3 4) # from 1 to VK_MAX_NCHANNELS, so if IMAGE_NCHANNELS changes - VK_MAX_NCHANNELS in defines.h should be updated
set(IMAGE_TYPES 8u 16u 32f)
set(TEMPLATE_DEFINES_NC_TYPES)
set(TEMPLATE_SUFFIXES_NC_TYPES)
set(TEMPLATE_DEFINES_TYPES)
set(TEMPLATE_SUFFIXES_TYPES)
foreach(type ${IMAGE_TYPES})
    string(TOUPPER ${type} type_upper)
    foreach(nchannels ${IMAGE_NCHANNELS})
        list(APPEND TEMPLATE_DEFINES_NC_TYPES "-DNCHANNELS=${nchannels} -DT_${type_upper}")
        list(APPEND TEMPLATE_SUFFIXES_NC_TYPES "${nchannels}x${type_upper}")
    endforeach(nchannels)
    list(APPEND TEMPLATE_DEFINES_TYPES "-DT_${type_upper}")
    list(APPEND TEMPLATE_SUFFIXES_TYPES "${type_upper}")
endforeach(type)

add_subdirectory(libgpu/vulkan) # it declares functions for Vulkan-related SPIR-V compilation

set(VULKAN_DEFINES)
compile_vulkan(HEADERS libgpu/vulkan/tests/kernels/aplusb.comp VULKAN_DEFINES)
compile_vulkan(HEADERS libgpu/vulkan/tests/kernels/atomic_add.comp VULKAN_DEFINES)
compile_vulkan(HEADERS libgpu/vulkan/tests/kernels/batched_binary_search.comp VULKAN_DEFINES)
compile_vulkan(HEADERS libgpu/vulkan/tests/kernels/rasterize.vert VULKAN_DEFINES)
compile_vulkan(HEADERS libgpu/vulkan/tests/kernels/rasterize.frag VULKAN_DEFINES)
compile_vulkan(HEADERS libgpu/vulkan/tests/kernels/rasterize_blending.frag VULKAN_DEFINES)
compile_vulkan(HEADERS libgpu/vulkan/tests/kernels/write_value_at_index.comp VULKAN_DEFINES)
compile_vulkan_templated(HEADERS libgpu/vulkan/tests/kernels/image_interpolation.comp VULKAN_DEFINES TEMPLATE_DEFINES_NC_TYPES TEMPLATE_SUFFIXES_NC_TYPES)
compile_vulkan_templated(HEADERS libgpu/vulkan/tests/kernels/image_conversion_from_float_to_T.comp VULKAN_DEFINES TEMPLATE_DEFINES_TYPES TEMPLATE_SUFFIXES_TYPES)

find_package(Vulkan REQUIRED)

if (GPU_CUDA_SUPPORT)
    enable_language(CUDA)
    find_package (CUDAToolkit REQUIRED)

    set(HEADERS ${HEADERS} ${CUDA_HEADERS})
    set(SOURCES ${SOURCES} ${CUDA_SOURCES})
    set(LIBRARIES ${LIBRARIES} CUDA::cuda_driver CUDA::cudart)

    add_definitions(-DCUDA_SUPPORT)
    add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
else ()
    add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
endif ()

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR} ${Vulkan_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})

add_executable (${PROJECT_NAME}_test ${HEADERS_TEST} ${SOURCES_TEST})
target_link_libraries (${PROJECT_NAME}_test ${LIBRARIES} ${GTEST_BOTH_LIBRARIES} ${CMAKE_DL_LIBS})
add_test(NAME ${PROJECT_NAME}_test COMMAND ${PROJECT_NAME}_test)
if (GPU_CUDA_SUPPORT)
    target_include_directories(${PROJECT_NAME}_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif ()
