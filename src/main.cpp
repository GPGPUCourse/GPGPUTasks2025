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

	// Таблица с кодами ошибок:
	// libs/clew/CL/cl.h:103
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

int main()
{
	// Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
	if(!ocl_init())
		throw std::runtime_error("Can't init OpenCL driver!");

	// Откройте
	// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
	// Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
	// Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
	std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

	// Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "    Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		const auto bad_param = static_cast<cl_platform_info>(239);
		size_t badParamSize = 0;
		cl_int err = clGetPlatformInfo(platform, bad_param, 0, nullptr, &badParamSize);
		std::string errString;
		switch (err)
		{
			case CL_SUCCESS: break;
			case CL_INVALID_VALUE: errString = "CL_INVALID_VALUE"; break;
			case CL_INVALID_PLATFORM: errString = "CL_INVALID_PLATFORM"; break;
			case CL_OUT_OF_HOST_MEMORY: errString = "CL_OUT_OF_HOST_MEMORY"; break;
		    default: errString = "Unknown error"; break;
		}
		if (err != CL_SUCCESS)
		{
			std::cout << "    OpenCL error code "
			          << to_string(err)
			          << " (" << errString << ")"
			          << " encountered at " << __FILE__ << ":" << to_string(__LINE__) <<std::endl;
		}

		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformName.size(), platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendor.size(), platformVendor.data(), nullptr));
		std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "Number of devices: " << devicesCount << std::endl;

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			cl_device_id device = devices[deviceIndex];

			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
			std::vector<unsigned char> deviceName(deviceNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceName.size(), deviceName.data(), nullptr));
			std::cout << "    Device name: " << deviceName.data() << std::endl;

			cl_device_type deviceType = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
			std::string deviceTypeString;
			switch(deviceType)
			{
				case CL_DEVICE_TYPE_CPU:
					deviceTypeString = "CPU";
					break;
				case CL_DEVICE_TYPE_GPU:
					deviceTypeString = "GPU";
					break;
				case CL_DEVICE_TYPE_ACCELERATOR:
					deviceTypeString = "Accelerator";
					break;
				case CL_DEVICE_TYPE_DEFAULT:
					deviceTypeString = "Default";
					break;
				default:
					deviceTypeString = "Unknown";
					break;
			}
			std::cout << "    Device type: " << deviceTypeString << std::endl;

			cl_ulong globalMem = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMem), &globalMem, nullptr));
			std::cout << "    Global memory size: " << (globalMem / (1024 * 1024)) << " MB" << std::endl;

			cl_uint computeUnits = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr));
			std::cout << "    Max compute units: " << computeUnits << std::endl;

			cl_uint maxClockFrequency = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFrequency), &maxClockFrequency, nullptr));
			std::cout << "    Max clock frequency: " << maxClockFrequency << " MHz" << std::endl;

			size_t maxWorkGroupSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr));
			std::cout << "    Max work group size: " << maxWorkGroupSize << std::endl;

			std::cout << std::endl;
		}
	}

	return 0;
}
