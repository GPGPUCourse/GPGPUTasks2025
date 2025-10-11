#include <CL/cl.h>
#include <assert.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

constexpr cl_ulong ONE_MB = 1024 * 1024;

template<typename T>
std::string to_string(T value)
{
	std::ostringstream ss;
	ss << value;
	return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
	if(CL_SUCCESS == err)
		return;

	// Таблица с кодами ошибок:
	// libs/clew/CL/cl.h:103
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

std::vector<unsigned char> readPlatformInfoString(cl_platform_id platform, cl_platform_info param_name)
{
	size_t size = 0;
	OCL_SAFE_CALL(clGetPlatformInfo(platform, param_name, 0, nullptr, &size));
	std::vector<unsigned char> result(size, 0);
	OCL_SAFE_CALL(clGetPlatformInfo(platform, param_name, size, result.data(), nullptr));

	return result;
}

std::vector<unsigned char> readDeviceInfoString(cl_device_id device_id, cl_device_info param_name)
{
	size_t size = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, param_name, 0, nullptr, &size));
	std::vector<unsigned char> result(size, 0);
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, param_name, size, result.data(), nullptr));

	return result;
}

template<typename T>
T readDeviceInfo(cl_device_id device_id, cl_device_info param_name)
{
	size_t size = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, param_name, 0, nullptr, &size));
	assert(size == sizeof(T));
	T result;
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, param_name, size, &result, nullptr));

	return result;
}

const char *deviceTypeToString(cl_device_type type)
{
	switch(type)
	{
	case CL_DEVICE_TYPE_DEFAULT:
		return "default";
	case CL_DEVICE_TYPE_CPU:
		return "cpu";
	case CL_DEVICE_TYPE_GPU:
		return "gpu";
	case CL_DEVICE_TYPE_ACCELERATOR:
		return "accel";
	case CL_DEVICE_TYPE_CUSTOM:
		return "custom";
	default:
		return "unknown";
	}
}

int main()
{
	if(!ocl_init())
		throw std::runtime_error("Can't init OpenCL driver!");

	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
	std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;
	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];
		std::cout << "    Platform name: " << readPlatformInfoString(platform, CL_PLATFORM_NAME).data() << std::endl;
		std::cout << "    Platform vendor: " << readPlatformInfoString(platform, CL_PLATFORM_VENDOR).data() << std::endl;

		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			cl_device_id dev = devices[deviceIndex];
			std::cout << "        Name: " << readDeviceInfoString(dev, CL_DEVICE_NAME).data() << std::endl;
			std::cout << "        Type: " << deviceTypeToString(readDeviceInfo<cl_device_type>(dev, CL_DEVICE_TYPE)) << std::endl;
			std::cout << "        Memory:  " << readDeviceInfo<cl_ulong>(dev, CL_DEVICE_GLOBAL_MEM_SIZE) / ONE_MB << std::endl;
			std::cout << "        Compute units:  " << readDeviceInfo<cl_uint>(dev, CL_DEVICE_MAX_COMPUTE_UNITS) << std::endl;
			std::cout << "        Kernels:  " << readDeviceInfoString(dev, CL_DEVICE_BUILT_IN_KERNELS).data() << std::endl;
		}
	}

	return 0;
}
