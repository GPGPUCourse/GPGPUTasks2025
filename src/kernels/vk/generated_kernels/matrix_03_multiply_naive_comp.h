#include "matrix_03_multiply_naive_comp_spirv_vulkan.h"

namespace avk2 {
	static const avk2::VersionedBinary* vulkan_versioned_binaries_matrix_03_multiply_naive_comp[] = {
		&vulkan_binary_matrix_03_multiply_naive_comp_spirv,
	};
	static ProgramBinaries vulkan_binaries_matrix_03_multiply_naive_comp = ProgramBinaries(std::vector<const VersionedBinary *>(vulkan_versioned_binaries_matrix_03_multiply_naive_comp, vulkan_versioned_binaries_matrix_03_multiply_naive_comp + sizeof(vulkan_versioned_binaries_matrix_03_multiply_naive_comp) / sizeof(vulkan_versioned_binaries_matrix_03_multiply_naive_comp[0])), "matrix_03_multiply_naive_comp");
}
