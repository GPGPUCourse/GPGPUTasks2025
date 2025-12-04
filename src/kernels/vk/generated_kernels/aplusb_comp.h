#include "aplusb_comp_spirv_vulkan.h"

namespace avk2 {
	static const avk2::VersionedBinary* vulkan_versioned_binaries_aplusb_comp[] = {
		&vulkan_binary_aplusb_comp_spirv,
	};
	static ProgramBinaries vulkan_binaries_aplusb_comp = ProgramBinaries(std::vector<const VersionedBinary *>(vulkan_versioned_binaries_aplusb_comp, vulkan_versioned_binaries_aplusb_comp + sizeof(vulkan_versioned_binaries_aplusb_comp) / sizeof(vulkan_versioned_binaries_aplusb_comp[0])), "aplusb_comp");
}
