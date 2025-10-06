#ifdef CUDA_SUPPORT
#include "libbase/timer.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include "enum.h"
#include "utils.h"

bool CUDAEnum::printInfo(int id, const std::string &opencl_driver_version)
{
	cudaError_t status;

	timer tm;

	cudaDeviceProp prop;
	CUDA_TRACE(status = cudaGetDeviceProperties(&prop, id));
	if (status != cudaSuccess)
		return false;

	int driverVersion = 239;
	CUDA_TRACE(status = cudaDriverGetVersion(&driverVersion));
	if (status != cudaSuccess)
		return false;

	int runtimeVersion = 239;
	CUDA_TRACE(status = cudaRuntimeGetVersion(&runtimeVersion));
	if (status != cudaSuccess)
		return false;

	double tm_properties = tm.elapsed();
	tm.restart();

	int previousCudaDevice = -1;
	CUDA_SAFE_CALL(cudaGetDevice(&previousCudaDevice));
	CUDA_SAFE_CALL(cudaSetDevice(id));
	size_t total_mem_size = 0;
	size_t free_mem_size = 0;
	CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem_size, &total_mem_size));
	CUDA_SAFE_CALL(cudaSetDevice(previousCudaDevice));

	double tm_memory = tm.elapsed();

	std::cout << "Using device: " << prop.name << ", " << prop.multiProcessorCount << " compute units, free memory: " << (free_mem_size >> 20) << "/" << (total_mem_size >> 20) << " MB, compute capability " <<  prop.major << "." << prop.minor << std::endl;
	if (!opencl_driver_version.empty()) {
		std::cout << "  driver version: " << opencl_driver_version << ", driver/runtime CUDA: " << driverVersion << "/" << runtimeVersion << std::endl;
	} else {
		std::cout << "  driver/runtime CUDA: " << driverVersion << "/" << runtimeVersion << std::endl;
	}
	std::cout << "  max work group size " << prop.maxThreadsPerBlock << std::endl;
	std::cout << "  max work item sizes [" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;

	double tm_total = tm_properties + tm_memory;
	if (tm_total > 0.5)
		std::cout << "  got device properties in " << tm_properties << " sec, free memory in " << tm_memory << " sec" << std::endl;

	return true;
}

CUDAEnum::CUDAEnum()
{
}

CUDAEnum::~CUDAEnum()
{
}

bool CUDAEnum::compareDevice(const Device &dev1, const Device &dev2)
{
	if (dev1.name	> dev2.name)	return false;
	if (dev1.name	< dev2.name)	return true;
	if (dev1.id		> dev2.id)		return false;
	return true;
}

bool CUDAEnum::enumDevices(bool silent)
{
	int device_count = 0;

	cudaError_t res = cudaSuccess;
	CUDA_TRACE(res = cudaGetDeviceCount(&device_count));
	if (res == cudaErrorNoDevice || res == cudaErrorInsufficientDriver)
		return true;

	if (res != cudaSuccess) {
		if (!silent) std::cerr << "cudaGetDeviceCount failed: " << cuda::formatError(res) << std::endl;
		return false;
	}

	for (int device_index = 0; device_index < device_count; device_index++) {
		cudaDeviceProp prop;

		CUDA_TRACE(res = cudaGetDeviceProperties(&prop, device_index));
		if (res != cudaSuccess) {
			if (!silent) std::cerr << "cudaGetDeviceProperties failed: " << cuda::formatError(res) << std::endl;
			return false;
		}

		// we don't support CUDA devices with compute capability < 2.0
		if (prop.major < 2)
			continue;

		Device device;

                int sm_clock_khz = 9999;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
                // CUDA 13+: runtime property removed -> query attribute (returns kHz)
                CUDA_SAFE_CALL( cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, device_index) );
#else
                // CUDA <= 12.x: still available in cudaDeviceProp (kHz)
                sm_clock_khz = prop.clockRate;
#endif

		device.id				= device_index;
		device.name				= prop.name;
		device.compute_units	= prop.multiProcessorCount;
		device.mem_size			= prop.totalGlobalMem;
		device.clock			= sm_clock_khz / 1000;
		device.pci_bus_id		= prop.pciBusID;
		device.pci_device_id	= prop.pciDeviceID;
		device.compcap_major	= prop.major;
		device.compcap_minor	= prop.minor;

		devices_.push_back(device);
	}

	std::sort(devices_.begin(), devices_.end(), compareDevice);

	return true;
}
#endif
