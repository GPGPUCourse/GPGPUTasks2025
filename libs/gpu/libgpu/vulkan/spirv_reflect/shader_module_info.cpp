#include "shader_module_info.h"

#include <libbase/string_utils.h>
#include <libbase/runtime_assert.h>

#include "spirv_reflect.h"

#include "../vulkan_api_headers.h"

namespace avk2 {
	class SpvReflectShaderModuleRAII {
	public:
		SpvReflectShaderModuleRAII(const SpvReflectShaderModule &module)
		 : module(module)
		{}

		virtual ~SpvReflectShaderModuleRAII()
		{
			spvReflectDestroyShaderModule(&module);
		}

		SpvReflectShaderModule *get()
		{
			return &module;
		}

	private:
		SpvReflectShaderModule module;
	};
}

avk2::ShaderModuleInfo::ShaderModuleInfo(const std::string &program_data, const std::string &name) {
	SpvReflectShaderModule shader_module = {};
	rassert(spvReflectCreateShaderModule(program_data.size(), (const uint32_t*) program_data.data(), &shader_module) == SPV_REFLECT_RESULT_SUCCESS, 653627099007097);
	shader_module_ = std::make_shared<SpvReflectShaderModuleRAII>(shader_module);
	name_ = name;
}

size_t avk2::ShaderModuleInfo::getPushConstantSize() const
{
	// check that there is a single push constant and get its size
	uint32_t push_constants_count = 0;
	rassert(spvReflectEnumeratePushConstantBlocks(shader_module_->get(), &push_constants_count, nullptr) == SPV_REFLECT_RESULT_SUCCESS, 35132421321312);
	rassert(push_constants_count <= 1, 898610934351210); // this is not a strict requirement, Vulkan supports multiple push constants, so in future we can support such cases
	if (push_constants_count == 0) { // this is often happens for rasterization (because work size is specified with viewport size), but nearly never happens for compute shaders (because push constants are used to specify work size)
		return 0;
	}
	std::vector<SpvReflectBlockVariable*> push_constants(push_constants_count);
	rassert(spvReflectEnumeratePushConstantBlocks(shader_module_->get(), &push_constants_count, push_constants.data()) == SPV_REFLECT_RESULT_SUCCESS, 35132421321312);
	uint32_t push_constant_size = push_constants[0][0].size; // note that it can be bigger than host-side push constant struct due to padding (f.e. 16 bytes if it is a single uint32)
	return push_constant_size;
}

size_t avk2::ShaderModuleInfo::getMergedPushConstantSize(const std::vector<ShaderModuleInfo> &shaders_module_info)
{
	rassert(shaders_module_info.size() >= 1, 4561536324);
	size_t merged_size = 0;
	for (size_t k = 0; k < shaders_module_info.size(); ++k) {
		size_t size = shaders_module_info[k].getPushConstantSize();
		if (size > 0) {
			if (merged_size == 0) {
				merged_size = size;
			} else {
				rassert(merged_size == size, 492551747);
			}
		}
	}
	return merged_size;
}

std::set<unsigned int> avk2::ShaderModuleInfo::getDescriptorsSets() const
{
	uint32_t bindings_count = 0;
	rassert(spvReflectEnumerateDescriptorBindings(shader_module_->get(), &bindings_count, nullptr) == SPV_REFLECT_RESULT_SUCCESS, 456621351241234);
	std::vector<SpvReflectDescriptorBinding*> descriptor_bindings(bindings_count);
	rassert(spvReflectEnumerateDescriptorBindings(shader_module_->get(), &bindings_count, descriptor_bindings.data()) == SPV_REFLECT_RESULT_SUCCESS, 7621351241234);

	std::set<unsigned int> sets;
	for (size_t i = 0; i < bindings_count; ++i) {
		sets.insert(descriptor_bindings[i]->set);
	}
	return sets;
}

std::set<unsigned int> avk2::ShaderModuleInfo::getMergedDescriptorsSets(const std::vector<ShaderModuleInfo> &shaders_module_info)
{
	rassert(shaders_module_info.size() >= 1, 992635631);
	// just union all sets
	std::set<unsigned int> merged_sets;
	for (size_t k = 0; k < shaders_module_info.size(); ++k) {
		std::set<unsigned int> sets = shaders_module_info[k].getDescriptorsSets();
		for (unsigned int set: sets) {
			merged_sets.insert(set);
		}
	}
	return merged_sets;
}

std::vector<std::optional<vk::DescriptorType>> avk2::ShaderModuleInfo::getDescriptorsTypes(unsigned int set) const
{
	uint32_t bindings_count = 0;
	rassert(spvReflectEnumerateDescriptorBindings(shader_module_->get(), &bindings_count, nullptr) == SPV_REFLECT_RESULT_SUCCESS, 4566213512416234);
	std::vector<SpvReflectDescriptorBinding*> descriptor_bindings(bindings_count);
	rassert(spvReflectEnumerateDescriptorBindings(shader_module_->get(), &bindings_count, descriptor_bindings.data()) == SPV_REFLECT_RESULT_SUCCESS, 76213571241234);

	for (size_t i = 0; i < bindings_count; ++i) {
		for (size_t j = i + 1; j < bindings_count; ++j) {
			rassert(descriptor_bindings[i]->binding != descriptor_bindings[j]->binding || descriptor_bindings[i]->set != descriptor_bindings[j]->set,
					"Two descriptors with the same binding=" + to_string(descriptor_bindings[i]->binding) + " and set=" + to_string(descriptor_bindings[i]->set) + " found, kernel=" + name_);
		}
	}

	size_t set_size = 0;
	std::set<size_t> used_set_bindings;
	for (size_t i = 0; i < bindings_count; ++i) {
		if (descriptor_bindings[i]->set == set) {
			size_t binding = descriptor_bindings[i]->binding;
			used_set_bindings.insert(binding);
			set_size = std::max(set_size, binding + 1);
		}
	}

	// check that set has N=set_size continuous bindings from 0 and up to (set_size-1)
	// TODO when it will make sense - remove this requirement with just skipping such bindings/args
	for (size_t binding = 0; binding < set_size; ++binding) {
		//rassert(used_set_bindings.count(binding), "kernel " + name_ + ": binding=", binding, " is missing (probably is unused in kernel), but bindings in range [0, ", set_size - 1, "] expected");
	}

	// rassert(set_size > 0, 837871549);
	// disabled due to typical situation: vert-shader has a CameraTransformVk (to project vertices into camera),
	// but frag-shader doesn't have any bindings

	std::vector<std::optional<vk::DescriptorType>> descriptor_types(set_size, std::nullopt);
	for (size_t i = 0; i < bindings_count; ++i) {
		if (descriptor_bindings[i]->set != set) {
			continue;
		}

		SpvReflectDescriptorType spv_type = descriptor_bindings[i]->descriptor_type;
		vk::DescriptorType type;
		if (spv_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
			type = vk::DescriptorType::eStorageBuffer;
		} else if (spv_type == SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
			type = vk::DescriptorType::eCombinedImageSampler;
		} else if (spv_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
			type = vk::DescriptorType::eStorageImage;
		} else {
			rassert(false, "unsupported descriptor type " + to_string(spv_type));
		}
		rassert((size_t) spv_type == (size_t) type, 840103752157962);

		unsigned int binding = descriptor_bindings[i]->binding;
		rassert(binding < set_size, 211025804); // we assume that bindings are consecutive from 0 up to (set_size - 1)
		descriptor_types[binding] = type;
	}
	for (auto descriptor_type: descriptor_types) {
		// rassert(descriptor_type.has_value(), 597119783); // TODO when it will fail - remove this requirement and add documentation about "when some type can be empty?"
	}
	return descriptor_types;
}

std::vector<std::optional<vk::DescriptorType>> avk2::ShaderModuleInfo::mergeDescriptorsTypes(const std::vector<std::vector<std::optional<vk::DescriptorType>>> &sets)
{
	size_t set_size = 0;
	for (size_t k = 0; k < sets.size(); ++k) {
		set_size = std::max(sets[k].size(), set_size);
	}
	std::vector<std::optional<vk::DescriptorType>> descriptor_types(set_size, std::nullopt);
	for (size_t k = 0; k < sets.size(); ++k) {
		if (sets[k].size() == 0) {
			// note that even if there are such uniform in shader code - if it is not used - it will not be presented in byte-code
			// but even more - it is typical that vert-shader has a CameraTransformVk (to project vertices into camera),
			// but frag-shader doesn't have any bindings
			continue;
		}
		for (size_t binding = 0; binding < set_size; ++binding) {
			if (binding < sets[k].size() && sets[k][binding].has_value()) {
				if (descriptor_types[binding].has_value()) {
					rassert(*descriptor_types[binding] == *sets[k][binding], 317745798);
				} else {
					descriptor_types[binding] = sets[k][binding];
				}
			}
		}
	}
	return descriptor_types;
}

std::vector<vk::DescriptorType> avk2::ShaderModuleInfo::getMergedDescriptorsTypes(const std::vector<ShaderModuleInfo> &shaders_module_info, unsigned int set)
{
	rassert(shaders_module_info.size() >= 1, 456555054);
	std::vector<std::vector<std::optional<vk::DescriptorType>>> sets(shaders_module_info.size());
	for (size_t k = 0; k < shaders_module_info.size(); ++k) {
		sets[k] = shaders_module_info[k].getDescriptorsTypes(set);
	}
	std::vector<std::optional<vk::DescriptorType>> merged = mergeDescriptorsTypes(sets);
	return ensureNoEmptyDescriptorTypes(merged); // TODO when it will fail - loosen the requirement and add documentation
}

std::vector<vk::DescriptorType> avk2::ShaderModuleInfo::ensureNoEmptyDescriptorTypes(const std::vector<std::optional<vk::DescriptorType>> &descriptor_types)
{
	std::vector<vk::DescriptorType> unpacked_descriptor_types(descriptor_types.size());
	for (size_t i = 0; i < descriptor_types.size(); ++i) {
		rassert(descriptor_types[i].has_value(), 637730197);
		unpacked_descriptor_types[i] = *descriptor_types[i];
	}
	return unpacked_descriptor_types;
}

bool avk2::ShaderModuleInfo::isDescriptorUsed(unsigned int set, unsigned int binding) const
{
	uint32_t bindings_count = 0;
	rassert(spvReflectEnumerateDescriptorBindings(shader_module_->get(), &bindings_count, nullptr) == SPV_REFLECT_RESULT_SUCCESS, 4566213512241234);
	std::vector<SpvReflectDescriptorBinding*> descriptor_bindings(bindings_count);
	rassert(spvReflectEnumerateDescriptorBindings(shader_module_->get(), &bindings_count, descriptor_bindings.data()) == SPV_REFLECT_RESULT_SUCCESS, 76213512541234);

	// TODO refactor - a lot of common logic with getDescriptorImageArrayDimsCount()
	bool binding_found = false;
	bool is_used = false;
	for (size_t i = 0; i < bindings_count; ++i) {
		if (descriptor_bindings[i]->set == set && descriptor_bindings[i]->binding == binding) {
			rassert(!binding_found, 114893068);
			binding_found = true;
			is_used = descriptor_bindings[i]->accessed;
		}
	}
	if (!binding_found) {
		return false;
	}

	return is_used;
}

bool avk2::ShaderModuleInfo::isImageArrayed(unsigned int set, unsigned int binding) const
{
	uint32_t bindings_count = 0;
	rassert(spvReflectEnumerateDescriptorBindings(shader_module_->get(), &bindings_count, nullptr) == SPV_REFLECT_RESULT_SUCCESS, 4566213512241234);
	std::vector<SpvReflectDescriptorBinding*> descriptor_bindings(bindings_count);
	rassert(spvReflectEnumerateDescriptorBindings(shader_module_->get(), &bindings_count, descriptor_bindings.data()) == SPV_REFLECT_RESULT_SUCCESS, 76213512541234);

	// TODO refactor - a lot of common logic with isDescriptorUsed()
	bool binding_found = false;
	unsigned int image_is_array = false;
	for (size_t i = 0; i < bindings_count; ++i) {
		if (descriptor_bindings[i]->set == set && descriptor_bindings[i]->binding == binding) {
			rassert(!binding_found, 114893062);
			binding_found = true;
			image_is_array = descriptor_bindings[i]->image.arrayed;
		}
	}
	rassert(binding_found, 134934522);

	return image_is_array;
}

bool avk2::ShaderModuleInfo::isDescriptorUsedInAny(const std::vector<ShaderModuleInfo> &shaders_module_info, unsigned int set, unsigned int binding)
{
	bool usage_found_at_least_in_one = false;
	for (size_t k = 0; k < shaders_module_info.size(); ++k) {
		if (shaders_module_info[k].isDescriptorUsed(set, binding)) {
			usage_found_at_least_in_one = true;
		}
	}
	return usage_found_at_least_in_one;
}

std::vector<size_t> avk2::ShaderModuleInfo::getGroupSize(const std::string &entry_point_name) const
{
	std::vector<size_t> group_size(3);
	const SpvReflectEntryPoint* entry_point = spvReflectGetEntryPoint(shader_module_->get(), entry_point_name.c_str());
	rassert(entry_point != nullptr, 987708471186706);
	group_size[0] = entry_point->local_size.x;
	group_size[1] = entry_point->local_size.y;
	group_size[2] = entry_point->local_size.z;
	return group_size;
}

SpvReflectBlockVariable avk2::ShaderModuleInfo::getBindingStructInfo(unsigned int set, unsigned int binding) const
{
	SpvReflectResult result_code;
	const SpvReflectDescriptorBinding* descriptor_binding = spvReflectGetDescriptorBinding(shader_module_->get(), binding, set, &result_code);
	rassert(result_code == SPV_REFLECT_RESULT_SUCCESS, 353734049);
	rassert(descriptor_binding, 937169278);

	return descriptor_binding->block;
}
