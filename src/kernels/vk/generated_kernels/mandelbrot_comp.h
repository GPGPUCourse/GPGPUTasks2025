#include "mandelbrot_comp_spirv_vulkan.h"

namespace avk2 {
	static const avk2::VersionedBinary* vulkan_versioned_binaries_mandelbrot_comp[] = {
		&vulkan_binary_mandelbrot_comp_spirv,
	};
	static ProgramBinaries vulkan_binaries_mandelbrot_comp = ProgramBinaries(std::vector<const VersionedBinary *>(vulkan_versioned_binaries_mandelbrot_comp, vulkan_versioned_binaries_mandelbrot_comp + sizeof(vulkan_versioned_binaries_mandelbrot_comp) / sizeof(vulkan_versioned_binaries_mandelbrot_comp[0])), "mandelbrot_comp");
}
