#include "sum_04_local_reduction_comp_spirv_vulkan.h"

namespace avk2 {
	static const avk2::VersionedBinary* vulkan_versioned_binaries_sum_04_local_reduction_comp[] = {
		&vulkan_binary_sum_04_local_reduction_comp_spirv,
	};
	static ProgramBinaries vulkan_binaries_sum_04_local_reduction_comp = ProgramBinaries(std::vector<const VersionedBinary *>(vulkan_versioned_binaries_sum_04_local_reduction_comp, vulkan_versioned_binaries_sum_04_local_reduction_comp + sizeof(vulkan_versioned_binaries_sum_04_local_reduction_comp) / sizeof(vulkan_versioned_binaries_sum_04_local_reduction_comp[0])), "sum_04_local_reduction_comp");
}
