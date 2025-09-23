#pragma once

#include <set>
#include <vector>
#include <memory>
#include <string>
#include <optional>

struct SpvReflectShaderModule;
typedef struct SpvReflectBlockVariable SpvReflectBlockVariable;

namespace vk {
	enum class DescriptorType;
}

namespace avk2 {

	class SpvReflectShaderModuleRAII;

	// using https://github.com/KhronosGroup/SPIRV-Reflect we can get info from kernel SPIR-V assembly:
	// - push constant size
	// - everything about descriptors
	//
	// It is useful also to run:
	// /opt/bin/spirv-reflect .../generated_kernels/aplusb_spirv_vulkan.spir
	//
	// How to extend this info?
	// 1) Find the relevant output line in spirv-reflect
	// 2) Open https://github.com/KhronosGroup/SPIRV-Reflect/blob/8406f76dcf6cca11fe430058c4f0ed4b846f3be4/common/output_stream.cpp
	// 3) Find that output line in that output_stream.cpp and find type <SomeType> of struct with relevant info
	// 4) Find relevant calls typing something like spvReflect<SomeType>
	class ShaderModuleInfo {
	public:
		ShaderModuleInfo() {}
		ShaderModuleInfo(const std::string &program_data, const std::string &name);

		size_t													getPushConstantSize() const;
		std::set<unsigned int>									getDescriptorsSets() const;
		std::vector<std::optional<vk::DescriptorType>>			getDescriptorsTypes(unsigned int set) const;
		bool													isDescriptorUsed(unsigned int set, unsigned int binding) const;
		bool													isImageArrayed(unsigned int set, unsigned int binding) const;
		SpvReflectBlockVariable									getBindingStructInfo(unsigned int set, unsigned int binding) const;
		std::vector<size_t>										getGroupSize(const std::string &entry_point_name) const;

		static size_t											getMergedPushConstantSize(const std::vector<ShaderModuleInfo> &shaders_module_info);
		static std::set<unsigned int>							getMergedDescriptorsSets(const std::vector<ShaderModuleInfo> &shaders_module_info);
		static std::vector<std::optional<vk::DescriptorType>>	mergeDescriptorsTypes(const std::vector<std::vector<std::optional<vk::DescriptorType>>> &sets);
		static std::vector<vk::DescriptorType>					getMergedDescriptorsTypes(const std::vector<ShaderModuleInfo> &shaders_module_info, unsigned int set);
		static std::vector<vk::DescriptorType>					ensureNoEmptyDescriptorTypes(const std::vector<std::optional<vk::DescriptorType>> &descriptor_types);
		static bool												isDescriptorUsedInAny(const std::vector<ShaderModuleInfo> &shaders_module_info, unsigned int set, unsigned int binding);

	protected:
		std::shared_ptr<SpvReflectShaderModuleRAII> shader_module_;
		std::string name_;
	};
}
