#pragma once

#include "utils.h"

#include "../../base/libbase/math.h"

#ifdef CUDA_SUPPORT
	#include <vector_types.h>
#endif

namespace gpu {
	class WorkSize {
	public:
		WorkSize(size_t groupSizeX, size_t workSizeX)
		{
			init(1, groupSizeX, 1, 1, workSizeX, 1, 1);
		}

		WorkSize(size_t groupSizeX, size_t groupSizeY, size_t workSizeX, size_t workSizeY)
		{
			init(2, groupSizeX, groupSizeY, 1, workSizeX, workSizeY, 1);
		}

		WorkSize(size_t groupSizeX, size_t groupSizeY, size_t groupSizeZ, size_t workSizeX, size_t workSizeY, size_t workSizeZ)
		{
			init(3, groupSizeX, groupSizeY, groupSizeZ, workSizeX, workSizeY, workSizeZ);
		}

#ifdef CUDA_SUPPORT
		const dim3 &cuBlockSize() const {
			return blockSize;
		}

		const dim3 &cuGridSize() const {
			return gridSize;
		}
#endif

		const size_t *clLocalSize() const {
			return localWorkSize;
		}

		const size_t *clGlobalSize() const {
			return globalWorkSize;
		}

		const size_t *vkGroupSize() const {
			return groupSize;
		}

		const size_t *vkGroupCount() const {
			return groupCount;
		}

		int clWorkDim() const {
			return workDims;
		}

	private:
		void init(int workDims, size_t groupSizeX, size_t groupSizeY, size_t groupSizeZ, size_t workSizeX, size_t workSizeY, size_t workSizeZ)
		{
			// round up workSize so that it is divisible by groupSize
			workSizeX = div_ceil(workSizeX, groupSizeX) * groupSizeX;
			workSizeY = div_ceil(workSizeY, groupSizeY) * groupSizeY;
			workSizeZ = div_ceil(workSizeZ, groupSizeZ) * groupSizeZ;

			// Vulkan
			groupSize[0] = groupSizeX;
			groupSize[1] = groupSizeY;
			groupSize[2] = groupSizeZ;

			groupCount[0] = workSizeX / groupSizeX;
			groupCount[1] = workSizeY / groupSizeY;
			groupCount[2] = workSizeZ / groupSizeZ;

			// OpenCL
			this->workDims = workDims;

			localWorkSize[0] = groupSizeX;
			localWorkSize[1] = groupSizeY;
			localWorkSize[2] = groupSizeZ;

			globalWorkSize[0] = workSizeX;
			globalWorkSize[1] = workSizeY;
			globalWorkSize[2] = workSizeZ;

#ifdef CUDA_SUPPORT
			// CUDA
			blockSize	= dim3((unsigned int) groupSizeX, (unsigned int) groupSizeY, (unsigned int) groupSizeZ);
			gridSize	= dim3((unsigned int) groupCount[0], (unsigned int) groupCount[1], (unsigned int) groupCount[2]);
#endif
		}

	private:
		// Vulkan
		size_t	groupSize[3];
		size_t	groupCount[3];

		// OpenCL
		size_t	localWorkSize[3];
		size_t	globalWorkSize[3];
		int		workDims;

		// CUDA
#ifdef CUDA_SUPPORT
		dim3	blockSize;
		dim3	gridSize;
#endif
	};
}