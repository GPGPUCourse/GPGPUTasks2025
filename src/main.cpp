#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

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
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

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

		// Получаем название платформы
		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// Получаем вендора платформы
		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
		std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

		// Получаем количество устройств
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Number of devices: " << devicesCount << std::endl;

		// Получаем информацию об устройствах
		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			cl_device_id device = devices[deviceIndex];

			// Название устройства
			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
			std::vector<unsigned char> deviceName(deviceNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
			std::cout << "      Device name: " << deviceName.data() << std::endl;

			// Тип устройства
			cl_device_type deviceType;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
			std::cout << "      Device type: ";
			if(deviceType & CL_DEVICE_TYPE_GPU) std::cout << "GPU";
			if(deviceType & CL_DEVICE_TYPE_CPU) std::cout << "CPU";
			if(deviceType & CL_DEVICE_TYPE_ACCELERATOR) std::cout << "Accelerator";
			std::cout << std::endl;

			// Размер памяти устройства
			cl_ulong globalMemSize;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMemSize, nullptr));
			std::cout << "      Global memory size: " << globalMemSize / (1024 * 1024) << " MB" << std::endl;

			// Максимальная частота устройства
			cl_uint maxClockFrequency;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &maxClockFrequency, nullptr));
			std::cout << "      Max clock frequency: " << maxClockFrequency << " MHz" << std::endl;
		}
	}

	return 0;
}
