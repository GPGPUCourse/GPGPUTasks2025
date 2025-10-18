# This CMakeLists.txt is used to declare functions for Vulkan-related SPIR-V compilation
cmake_minimum_required (VERSION 3.20)

# returns file name with neither the directory nor the longest extension + _comp/_vert/_frag suffix
# example: program_name=gpu/vulkan/tests/kernels/aplusb.comp => program_name=aplusb and ext_name=_comp
function(get_program_name program program_name ext_name)
	get_filename_component(VK_FILENAME ${program} NAME_WE)
	get_filename_component(VK_EXT ${program} EXT)
	string(REPLACE "." "" VK_EXT ${VK_EXT}) # removing the dot (first symbol) from extension
	set(${program_name} "${VK_FILENAME}" PARENT_SCOPE)
	if (VK_EXT STREQUAL "vert" OR VK_EXT STREQUAL "frag")
		set(${ext_name} "_${VK_EXT}" PARENT_SCOPE)
	elseif (VK_EXT STREQUAL "comp")
		set(${ext_name} "_comp" PARENT_SCOPE)
	else ()
		message(SEND_ERROR "Vulkan shaders only with .vert/.frag/.comp are supported, but ${program} with .${VK_EXT} extension found")
	endif()
endfunction()

function(join_defines VALUES OUTPUT)
	string (REPLACE " " "_" _TMP_STR "${${VALUES}}")
	string (REPLACE "-D" "_" _TMP_STR ${_TMP_STR})
	string (REPLACE "=" "_" _TMP_STR ${_TMP_STR})
	string (REGEX REPLACE "__+" "_" _TMP_STR ${_TMP_STR})
	string (REGEX REPLACE "^_+$" "" _TMP_STR ${_TMP_STR})
	set (${OUTPUT} "${_TMP_STR}" PARENT_SCOPE)
endfunction()

function (compile_vulkan_spir outfiles program defines)
	get_program_name (${program} VK_NAME EXT_NAME)
	get_filename_component (VK_DIR ${program} DIRECTORY) # Directory without file name
	set (VK_DIR "${VK_DIR}/generated_kernels")
	if (NOT "${${defines}}" STREQUAL "")
		join_defines(${defines} defines_str)
	endif()

	set (VARIABLE_SUFFIX ${VK_NAME}${defines_str}${EXT_NAME}_spirv)
	set (PREPROCESSED_SOURCE	${PROJECT_SOURCE_DIR}/${VK_DIR}/${VARIABLE_SUFFIX}_vulkan.c)
	set (INCLUDE_DEPENDENCIES	${PROJECT_SOURCE_DIR}/${VK_DIR}/${VARIABLE_SUFFIX}_vulkan.d) # this file is generated thanks to glslc -MD -MF flags and makes possible to have automatical track of dependencies changes (using DEPFILE)
	set (SPIR_BINARY			${PROJECT_SOURCE_DIR}/${VK_DIR}/${VARIABLE_SUFFIX}_vulkan.spir)
	set (VK_HEADER				${PROJECT_SOURCE_DIR}/${VK_DIR}/${VARIABLE_SUFFIX}_vulkan.h) # it will include avk::VersionedBinary

	set (GLSLC_DEBUG_FLAG "")
	if (CMAKE_BUILD_TYPE MATCHES "Debug")
		set (GLSLC_DEBUG_FLAG "-g")
	endif()

	separate_arguments(defines_separated UNIX_COMMAND ${${defines}})
	if (NOT CMAKE_BUILD_TYPE MATCHES "Debug")
		list(APPEND defines_separated "-DNDEBUG") # we rely on NDEBUG in common.vk on device-side (to enable/disable DEBUG_PRINTF_EXT_ENABLED)
	endif ()

	# Subdivision into two add_custom_command(...) is required
	# because we need to use DEPFILE (and thus cmake>=3.20) to have an automatic tracking of #include-based dependencies changes
	# and it doesn't work (don't know why) if we specify ${VK_HEADER} as OUTPUT instead of ${SPIR_BINARY}.
	# On the other hand, ${VK_HEADER} should be specified as OUTPUT
	# so that CMake will know that "those headers that are used later - are generated with this add_custom_command",
	# so we split this task into chain of two add_custom_command:
	# 1) compile shader kernel into SPIR-V assembly binary file (with proper dependencies changes tracking thanks to DEPFILE and glslc -MD -MF flags)
	# 2) convert SPIR-V assembly into C-array in header (with proper DEPENDS on SPIR-V assembly binary file)
	#
	# About DEPFILE see here:
	# - https://stackoverflow.com/questions/71003674/using-glslcs-depfile-to-make-included-files-automatically-trigger-recompile-of
	# - https://stackoverflow.com/questions/60420700/cmake-invocation-of-glslc-with-respect-to-includes-dependencies
	add_custom_command (
		OUTPUT ${SPIR_BINARY}
		DEPENDS ${program}
		DEPFILE ${INCLUDE_DEPENDENCIES} # this file has a list of direct and indirect dependencies (through #include), when they changes - this kernel should be re-compiled

		COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/${VK_DIR}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/${VK_DIR} # mkdir generated_kernels

		# compile shader kernel into SPIR-V assembly binary file
		COMMAND ${GLSLC_BIN} -MD -MF ${INCLUDE_DEPENDENCIES} -I${libgpu_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/${program} ${defines_separated} ${GLSLC_DEBUG_FLAG} -O -mfmt=bin -o ${SPIR_BINARY}
	)
	add_custom_command (
		OUTPUT ${VK_HEADER}
		DEPENDS hexdumparray ${SPIR_BINARY}

		COMMAND ${CMAKE_COMMAND} -E remove ${VK_HEADER}

		COMMAND ${CMAKE_COMMAND} -E echo \#include<libgpu/vulkan/engine.h> >> ${VK_HEADER}
		COMMAND ${CMAKE_COMMAND} -E echo namespace avk2 { >> ${VK_HEADER}

		# and convert SPIR-V assembly binary file into C-array in header
		COMMAND hexdumparray ${SPIR_BINARY} ${VK_HEADER} vulkan_${VARIABLE_SUFFIX}

		COMMAND ${CMAKE_COMMAND} -E echo static const VersionedBinary vulkan_binary_${VARIABLE_SUFFIX}\(vulkan_${VARIABLE_SUFFIX}, sizeof\(vulkan_${VARIABLE_SUFFIX}\)\)\; >> ${VK_HEADER}

		COMMAND ${CMAKE_COMMAND} -E echo } >> ${VK_HEADER}
		VERBATIM
	)

	set (${outfiles} ${${outfiles}} ${VK_HEADER} PARENT_SCOPE)
endfunction (compile_vulkan_spir)

function (compile_vulkan outfiles program defines)
	compile_vulkan_spir(${outfiles} ${program} ${defines})

	get_program_name (${program} VK_NAME EXT_NAME)
	get_filename_component (VK_DIR ${program} DIRECTORY)
	set(VK_DIR "${VK_DIR}/generated_kernels")
	if (NOT "${${defines}}" STREQUAL "")
		join_defines(${defines} defines_str)
	endif()
	set(PROGRAM_NAME ${VK_NAME}${defines_str}${EXT_NAME})
	set(VK_HEADER ${PROJECT_SOURCE_DIR}/${VK_DIR}/${PROGRAM_NAME}.h)
	configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/vk/vulkan_program.h.in ${VK_HEADER})
	set (${outfiles} ${${outfiles}} ${VK_HEADER} PARENT_SCOPE)
endfunction (compile_vulkan)

function (compile_vulkan_templated outfiles program defines template_defines template_suffixes)
	foreach(template_define template_suffix IN ZIP_LISTS ${template_defines} ${template_suffixes})
		set(defines_with_template_define "${${defines}} ${template_define}")
		compile_vulkan(${outfiles} ${program} defines_with_template_define)
	endforeach()

	get_program_name (${program} VK_NAME EXT_NAME)
	get_filename_component (VK_DIR ${program} DIRECTORY)
	set(VK_DIR "${VK_DIR}/generated_kernels")
	set(PROGRAM_NAME ${VK_NAME}${defines_str}${EXT_NAME})
	set(VK_HEADER ${PROJECT_SOURCE_DIR}/${VK_DIR}/${PROGRAM_NAME}.h)

	set(HEADER_WITH_TEMPLATED_MAP "// generated automatically from CMake function compile_vulkan_templated\n")
	foreach(template_define ${${template_defines}})
		join_defines(template_define defines_str)
		string(APPEND HEADER_WITH_TEMPLATED_MAP "#include \"${VK_NAME}${defines_str}${EXT_NAME}.h\"\n")
	endforeach()
	string(APPEND HEADER_WITH_TEMPLATED_MAP "#include <unordered_map>\n")
	string(APPEND HEADER_WITH_TEMPLATED_MAP "namespace avk2 {\n")
	string(APPEND HEADER_WITH_TEMPLATED_MAP "std::unordered_map<std::string, const avk2::ProgramBinaries*> init_map_${VK_NAME}()\n")
	string(APPEND HEADER_WITH_TEMPLATED_MAP "{\n")
	string(APPEND HEADER_WITH_TEMPLATED_MAP "	std::unordered_map<std::string, const avk2::ProgramBinaries*> templated_kernels_map\;\n")
	foreach(template_define template_suffix IN ZIP_LISTS ${template_defines} ${template_suffixes})
		set(defines_with_template_define "${${defines}} ${template_define}")
		join_defines(defines_with_template_define defines_str)
		string(APPEND HEADER_WITH_TEMPLATED_MAP "	templated_kernels_map[\"${template_suffix}\"] = &vulkan_binaries_${VK_NAME}${defines_str}${EXT_NAME}\;\n")
	endforeach()
	string(APPEND HEADER_WITH_TEMPLATED_MAP "	return templated_kernels_map\;\n")
	string(APPEND HEADER_WITH_TEMPLATED_MAP "}\n")
	string(APPEND HEADER_WITH_TEMPLATED_MAP "static std::unordered_map<std::string, const avk2::ProgramBinaries*> vulkan_binaries_${VK_NAME} = init_map_${VK_NAME}()\;\n")
	string(APPEND HEADER_WITH_TEMPLATED_MAP "}\n")
	file(WRITE ${VK_HEADER} ${HEADER_WITH_TEMPLATED_MAP})

	set (${outfiles} ${${outfiles}} ${VK_HEADER} PARENT_SCOPE)
endfunction (compile_vulkan_templated)
