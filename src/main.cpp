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
namespace {
	std::string prettify_mem(cl_uint mem_in_bytes) {
		std::vector<std::string> postfixes = {"B", "KB", "MB", "GB", "TB", "PT", "ZT"};
		size_t postfixes_idx = 0;

		while (mem_in_bytes > (1 << 16) && postfixes_idx + 1 < postfixes.size()) {
			mem_in_bytes /= 1024;
			postfixes_idx += 1;
		}
		return to_string(mem_in_bytes) + " " + postfixes[postfixes_idx];
	};

} // anonymous namespace

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
		// Answer: заменил на 1337, получил -30 с core dumped
		// Это - CL_INVALID_VALUE на месте libs/clew/CL/cl.h:204

		std::vector<char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformName.size(), platformName.data(), nullptr));
		std::cout << "\tPlatform name: " << platformName.data() << std::endl;

		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));
		std::vector<char> vendorName(vendorNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorName.size(), vendorName.data(), nullptr));
		std::cout << "\tVendor name: " << vendorName.data() << std::endl;

		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::vector<cl_device_id> devicesIds(devicesCount, 0);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesIds.size(), devicesIds.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			std::cout << "\tDevice #" << deviceIndex + 1 << "/" << devicesCount << std::endl;

			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devicesIds[deviceIndex], CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
			std::vector<char> deviceName(deviceNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(devicesIds[deviceIndex], CL_DEVICE_NAME, deviceName.size(), deviceName.data(), 0));
			std::cout << "\t\tDevice name: " << deviceName.data() << std::endl;

			cl_device_type deviceType;
			OCL_SAFE_CALL(clGetDeviceInfo(devicesIds[deviceIndex], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
			std::string deviceTypeStr;

			if (deviceType & CL_DEVICE_TYPE_CPU) { deviceTypeStr += "CPU "; }
			if (deviceType & CL_DEVICE_TYPE_GPU) { deviceTypeStr += "GPU "; }
			if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) { deviceTypeStr += "ACCELERATOR "; }
			if (deviceType & CL_DEVICE_TYPE_DEFAULT) { deviceTypeStr += "DEFAULT "; }
			if (deviceType & CL_DEVICE_TYPE_CUSTOM) { deviceTypeStr += "CUSTOM "; }
			if (deviceTypeStr.empty()) { deviceTypeStr = "Unknown"; }

			std::cout << "\t\tDevice type: " << deviceTypeStr << std::endl;

			cl_ulong deviceMemory;
			OCL_SAFE_CALL(clGetDeviceInfo(devicesIds[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceMemory), &deviceMemory, nullptr));
			std::cout << "\t\tDevice memory size: " << prettify_mem(deviceMemory) << std::endl;

			cl_ulong deviceCacheSize{};
			OCL_SAFE_CALL(clGetDeviceInfo(devicesIds[deviceIndex], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(deviceCacheSize), &deviceCacheSize, nullptr));
			std::cout << "\t\tDevice cache memory size: " << prettify_mem(deviceCacheSize) << std::endl;

			cl_bool deviceIsAvaliable{};
			OCL_SAFE_CALL(clGetDeviceInfo(devicesIds[deviceIndex], CL_DEVICE_AVAILABLE, sizeof(deviceIsAvaliable), &deviceIsAvaliable, nullptr));
			std::cout << "\t\tDevice is available: " << (deviceIsAvaliable ? "true" : "false") << std::endl;

			cl_uint deviceMaxClockFrequency{};
			OCL_SAFE_CALL(clGetDeviceInfo(devicesIds[deviceIndex], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(deviceMaxClockFrequency), &deviceMaxClockFrequency, nullptr));
			std::cout << "\t\tDevice clock: " << deviceMaxClockFrequency << " MHz" << std::endl;
		}
	}

	return 0;
}
