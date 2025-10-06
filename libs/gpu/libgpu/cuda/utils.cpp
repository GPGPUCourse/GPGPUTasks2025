#ifdef CUDA_SUPPORT
#include "utils.h"
#include "cuda_api.h"

#define CUDA_KERNELS_ACCURATE_ERRORS_CHECKS true

#ifndef NDEBUG
#undef CUDA_KERNELS_ACCURATE_ERRORS_CHECKS
#define CUDA_KERNELS_ACCURATE_ERRORS_CHECKS true
#endif

namespace cuda {

std::string formatError(cudaError_t code)
{
	return std::string(cudaGetErrorString(code)) + " (" + to_string(code) + ")";
}

void reportError(cudaError_t err, int line, const std::string &prefix)
{
	if (cudaSuccess == err)
		return;

	std::string message = prefix + formatError(err) + " at line " + to_string(line);

	size_t total_mem_size = 0;
	size_t free_mem_size = 0;
	cudaError_t err2;

	switch (err) {
	case cudaErrorMemoryAllocation:
		err2 = cudaMemGetInfo(&free_mem_size, &total_mem_size);
		if (cudaSuccess == err2)
			message = message + "(free memory: " + to_string(free_mem_size >> 20) + "/" + to_string(total_mem_size >> 20) + " MB)";
		else
			message = message + "(free memory unknown: " + formatError(err2) + ")";
		throw cuda_bad_alloc(message);
	default:
		throw cuda_exception(message);
	}
}

void checkKernelErrors(cudaStream_t stream, int line, bool synchronized)
{
	reportError(cudaGetLastError(), line, "Kernel failed: ");
	if (synchronized) {
		reportError(cudaStreamSynchronize(stream), line, "Kernel failed: ");
	}
}

void checkKernelErrors(cudaStream_t stream, int line)
{
	checkKernelErrors(stream, line, CUDA_KERNELS_ACCURATE_ERRORS_CHECKS);
}

}

#endif
