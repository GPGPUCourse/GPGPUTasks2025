#include "merge_sort_comp_spirv_vulkan.h"

namespace avk2 {
	static const avk2::VersionedBinary* vulkan_versioned_binaries_merge_sort_comp[] = {
		&vulkan_binary_merge_sort_comp_spirv,
	};
	static ProgramBinaries vulkan_binaries_merge_sort_comp = ProgramBinaries(std::vector<const VersionedBinary *>(vulkan_versioned_binaries_merge_sort_comp, vulkan_versioned_binaries_merge_sort_comp + sizeof(vulkan_versioned_binaries_merge_sort_comp) / sizeof(vulkan_versioned_binaries_merge_sort_comp[0])), "merge_sort_comp");
}
