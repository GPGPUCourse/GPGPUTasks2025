#include "kernels.h"

#include "generated_kernels/aplusb_comp.h"
#include "generated_kernels/atomic_add_comp.h"
#include "generated_kernels/batched_binary_search_comp.h"
#include "generated_kernels/image_conversion_from_float_to_T_comp.h"
#include "generated_kernels/image_interpolation_comp.h"
#include "generated_kernels/rasterize_frag.h"
#include "generated_kernels/rasterize_vert.h"
#include "generated_kernels/rasterize_blending_frag.h"
#include "generated_kernels/write_value_at_index_comp.h"

#include <libgpu/vulkan/vk/common_host.h>

namespace avk2 {
	const ProgramBinaries& getAplusBKernel() {
		return vulkan_binaries_aplusb_comp;
	}
	const ProgramBinaries& getAtomicAddKernel() {
		return vulkan_binaries_atomic_add_comp;
	}
	const ProgramBinaries& getBatchedBinarySearch() {
		return vulkan_binaries_batched_binary_search_comp;
	}
	const ProgramBinaries& getImageConversionFromFloatToT(DataType type) {
		rassert(type == DataType8u || type == DataType16u || type == DataType32f, 5462312324, typeName(type));
		std::string type_name = toupper(typeName(type));
		rassert(vulkan_binaries_image_conversion_from_float_to_T.count(type_name) > 0, 4613543412, type_name);
		return *vulkan_binaries_image_conversion_from_float_to_T[type_name];
	}
	const ProgramBinaries& getInterpolationKernel(DataType type, int nchannels) {
		rassert(type == DataType8u || type == DataType16u || type == DataType32f, 251615253, typeName(type));
		rassert(nchannels >= 1 && nchannels <= VK_MAX_NCHANNELS, 7348921634, nchannels);
		std::string type_name = to_string(nchannels) + "x" + toupper(typeName(type));
		rassert(vulkan_binaries_image_interpolation.count(type_name) > 0, 770673198, type_name);
		return *vulkan_binaries_image_interpolation[type_name];
	}
	std::vector<const ProgramBinaries*> getRasterizeKernel() {
		return {&vulkan_binaries_rasterize_vert, &vulkan_binaries_rasterize_frag};
	}
	std::vector<const ProgramBinaries*> getRasterizeWithBlendingKernel() {
		return {&vulkan_binaries_rasterize_vert, &vulkan_binaries_rasterize_blending_frag};
	}
	const ProgramBinaries& getWriteValueAtIndexKernel() {
		return vulkan_binaries_write_value_at_index_comp;
	}
}
