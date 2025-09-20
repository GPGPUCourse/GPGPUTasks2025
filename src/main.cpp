#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
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

template<typename T>
std::vector<T> getDeviceProp(cl_device_id devId, cl_device_info devProp)
{
	size_t propSize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(devId, devProp, 0, nullptr, &propSize));
	std::vector<T> propData(propSize);

	OCL_SAFE_CALL(clGetDeviceInfo(devId, devProp, propSize, propData.data(), nullptr));
	return propData;
}

std::string getDeviceTypeString(cl_device_type deviceType)
{
	std::string typeString;
	if(deviceType & CL_DEVICE_TYPE_CPU)
	{
		typeString += "CL_DEVICE_TYPE_CPU | ";
	}
	if(deviceType & CL_DEVICE_TYPE_GPU)
	{
		typeString += "CL_DEVICE_TYPE_GPU | ";
	}
	if(deviceType & CL_DEVICE_TYPE_ACCELERATOR)
	{
		typeString += "CL_DEVICE_TYPE_ACCELERATOR | ";
	}
	if(deviceType & CL_DEVICE_TYPE_DEFAULT)
	{
		typeString += "CL_DEVICE_TYPE_DEFAULT | ";
	}
	if(deviceType & CL_DEVICE_TYPE_CUSTOM)
	{
		typeString += "CL_DEVICE_TYPE_CUSTOM | ";
	}
	if(deviceType & CL_DEVICE_TYPE_ALL)
	{
		typeString += "CL_DEVICE_TYPE_ALL | ";
	}
	return typeString.empty() ? "UNKNOWN_DEVICE_TYPE" : typeString;
}

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

		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));

		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));
		std::vector<unsigned char> vendorName(vendorNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize, vendorName.data(), nullptr));
		std::cout << "    Vendor name: " << platformName.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Available devices count: " << devicesCount << std::endl;

		std::vector<cl_device_id> deviceIds(devicesCount, 0);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, deviceIds.data(), nullptr));

		for(size_t deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			auto cur_device = deviceIds[deviceIndex];
			auto deviceName = getDeviceProp<unsigned char>(cur_device, CL_DEVICE_NAME);
			auto devType = getDeviceProp<cl_device_type>(cur_device, CL_DEVICE_TYPE).at(0);
			auto maxMemoryBytes = getDeviceProp<cl_ulong>(cur_device, CL_DEVICE_MAX_MEM_ALLOC_SIZE).at(0);

			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			auto maxComputeUnits = getDeviceProp<cl_uint>(cur_device, CL_DEVICE_MAX_COMPUTE_UNITS).at(0);
			auto deviceVersion = getDeviceProp<char>(cur_device, CL_DEVICE_VERSION);

			std::cout << "		Device name: " << deviceName.data() << std::endl;
			std::cout << "		Device type: " << getDeviceTypeString(devType) << std::endl;
			std::cout << "		Max memory (MB): " << maxMemoryBytes / 1024 / 1024 << std::endl;
			std::cout << "		Max compute units: " << maxComputeUnits << std::endl;
			std::cout << "		Device version: " << deviceVersion.data() << std::endl;
		}
	}

	return 0;
}
