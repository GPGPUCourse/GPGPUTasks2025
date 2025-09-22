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

std::string get_device_type(cl_device_type deviceType);

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

		// ANS: Заменив на `OCL_SAFE_CALL(clGetPlatformInfo(platform, 0xBEEF, 0, nullptr, &platformNameSize));` получил код ошибки -30, что соответствует CL_INVALID_VALUE

		// Название платформы
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// Вендор платформы
		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, std::addressof(platformVendorSize)));
		std::vector<unsigned char> vendorName(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, vendorName.data(), nullptr));
		std::cout << "    Vendor name: " << vendorName.data() << std::endl;

		// Число доступных устройств данной платформы
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, std::addressof(devicesCount)));
		std::vector<cl_device_id> deviceIDs(devicesCount, 0);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, deviceIDs.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			cl_device_id device = deviceIDs[deviceIndex];

			// - Название устройства
			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, std::addressof(deviceNameSize)));
			std::vector<unsigned char> deviceName(deviceNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
			std::cout << "        Device name: " << deviceName.data() << std::endl;

			// - Тип устройства (видеокарта/процессор/что-то странное)
			cl_device_type deviceType;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), std::addressof(deviceType), nullptr));
			std::string deviceTypeName = get_device_type(deviceType);
			std::cout << "        Device type: " << deviceTypeName << std::endl;

			// - Размер памяти устройства в мегабайтах
			cl_ulong deviceGlobalMemSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceGlobalMemSize), std::addressof(deviceGlobalMemSize), nullptr));
			std::cout << "        Device global memory size: " << (deviceGlobalMemSize / (1024 * 1024)) << " MB" << std::endl;

			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			cl_uint computeUnits = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), std::addressof(computeUnits), nullptr));
			std::cout << "        Device max compute units: " << computeUnits << std::endl;
			cl_uint maxClockFrequency = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFrequency), std::addressof(maxClockFrequency), nullptr));
			std::cout << "        Device max clock frequency: " << maxClockFrequency << " MHz" << std::endl;
			cl_uint addressBits = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(addressBits), std::addressof(addressBits), nullptr));
			std::cout << "        Device address bits: " << addressBits << std::endl;
			cl_ulong localMemSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), std::addressof(localMemSize), nullptr));
			std::cout << "        Device local memory size: " << (localMemSize / 1024) << " KB" << std::endl;
		}
	}

	return 0;
}

std::string get_device_type(cl_device_type deviceType)
{
	std::string deviceTypeName;
	for(auto [mask, type] : { std::pair{ CL_DEVICE_TYPE_CPU, "CPU" },
	                          { CL_DEVICE_TYPE_GPU, "GPU" },
	                          { CL_DEVICE_TYPE_ACCELERATOR, "ACCELERATOR" },
	                          { CL_DEVICE_TYPE_DEFAULT, "DEFAULT" },
	                          { CL_DEVICE_TYPE_CUSTOM, "CUSTOM" } })
	{
		if(deviceType & mask)
		{
			deviceTypeName += type;
			deviceTypeName += " ";
		}
	}

	if(deviceTypeName.empty())
	{
		return "[Unknown type]";
	}
	return deviceTypeName;
}
