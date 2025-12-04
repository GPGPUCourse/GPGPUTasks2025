#include "aplusb_matrix_bad_comp_spirv_vulkan.h"

namespace avk2 {
	static const avk2::VersionedBinary* vulkan_versioned_binaries_aplusb_matrix_bad_comp[] = {
		&vulkan_binary_aplusb_matrix_bad_comp_spirv,
	};
	static ProgramBinaries vulkan_binaries_aplusb_matrix_bad_comp = ProgramBinaries(std::vector<const VersionedBinary *>(vulkan_versioned_binaries_aplusb_matrix_bad_comp, vulkan_versioned_binaries_aplusb_matrix_bad_comp + sizeof(vulkan_versioned_binaries_aplusb_matrix_bad_comp) / sizeof(vulkan_versioned_binaries_aplusb_matrix_bad_comp[0])), "aplusb_matrix_bad_comp");
}
