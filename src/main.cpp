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
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];
		
		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));

		// OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize));
		// Номер ошибки: -30
		// Название ошибки: CL_INVALID_VALUE
		// Описание ошибки:
		// CL_INVALID_VALUE if param_name is not one of the supported values or if size in bytes specified 
		// by param_value_size is less than size of return type and param_value is not a NULL value.

		
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		// clGetPlatformInfo(...);
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));
		
		
		std::vector<unsigned char> vendorName(vendorNameSize);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize, vendorName.data(), nullptr));
		// clGetPlatformInfo(...);
		std::cout << "    Vendor name: " << vendorName.data() << std::endl;
		
		
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		
		std::cout << "Number of OpenCL available devices: " << devicesCount << std::endl;
		
		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
		
		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			std::cout << "Platform #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
			
			std::vector<unsigned char> deviceName(deviceNameSize);
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
			std::cout << "    Device name: " << deviceName.data() << std::endl;

			cl_device_type deviceType = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
			
			std::string deviceTypeName;
			if (deviceType && CL_DEVICE_TYPE_CPU) {
				deviceTypeName = "CPU";
			} else if (deviceType && CL_DEVICE_TYPE_GPU) {
				deviceTypeName = "GPU";
			} else {
				deviceTypeName = "Something strange";
			}
			
			std::cout << "    Device type name: " << deviceTypeName << std::endl;
			
			cl_ulong deviceGlobalMemorySize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceGlobalMemorySize), &deviceGlobalMemorySize, nullptr));
			std::cout << "    Device global memory size: " << deviceGlobalMemorySize / (1000 * 1000) << " Mb" << std::endl; 

			cl_ulong deviceGlobalMemoryCacheSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(deviceGlobalMemoryCacheSize), &deviceGlobalMemoryCacheSize, nullptr));
			std::cout << "    Device global memory cache size: " << deviceGlobalMemoryCacheSize / 1000 << " Kb" << std::endl; 

			cl_ulong deviceGlobalMemoryCachelineSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(deviceGlobalMemoryCachelineSize), &deviceGlobalMemoryCachelineSize, nullptr));
			std::cout << "    Device global memory cacheline size: " << deviceGlobalMemoryCachelineSize << " b" << std::endl; 
			
		}
	}

	return 0;
}
