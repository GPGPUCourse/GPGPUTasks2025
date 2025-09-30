#pragma once

#include <libgpu/vulkan/engine.h>

namespace avk2 {
	const ProgramBinaries& getAplusBKernel();
	const ProgramBinaries& getAtomicAddKernel();
	const ProgramBinaries& getBatchedBinarySearch();
	const ProgramBinaries& getImageConversionFromFloatToT(DataType type);
	const ProgramBinaries& getInterpolationKernel(DataType type, int nchannels);
	std::vector<const ProgramBinaries*> getRasterizeKernel();
	std::vector<const ProgramBinaries*> getRasterizeWithBlendingKernel();
	const ProgramBinaries& getWriteValueAtIndexKernel();
}
