#include "CL/cl_platform.h"
#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <cstddef>
#include <iostream>
#include <optional>
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

// I despise repeating code
// I despise non-uniform code
// I am too lazy to rewrite this into classes so we stay c-style
namespace CL {
cl_uint GetPlatformsCount() {
	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
	return platformsCount;
}

std::vector<cl_platform_id> GetPlatformIds(std::optional<cl_uint> hint = std::nullopt) {
	cl_uint platformsCount = hint.value_or(GetPlatformsCount());
	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));
	return platforms;
}

std::string GetPlatformInfo(cl_platform_id platform, cl_platform_info paramName) {
	size_t platformNameSize = 0;
	OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));

	std::vector<char> platformName(platformNameSize, 0);
	OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));

	return std::string(platformName.data());
}

cl_uint GetDevicesCount(cl_platform_id platform, cl_device_type deviceType) {
	cl_uint devicesCount = 0;
	OCL_SAFE_CALL(clGetDeviceIDs(platform, deviceType, 0, nullptr, &devicesCount));
	return devicesCount;
}

std::vector<cl_device_id> GetDeviceIds(cl_platform_id platform, cl_device_type deviceType, std::optional<cl_uint> hint = std::nullopt) {
	cl_uint devicesCount = hint ? *hint : GetDevicesCount(platform, deviceType);
	std::vector<cl_device_id> devices(devicesCount);
	OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
	return devices;
}

template <typename T>
T GetDeviceInfo(cl_device_id device, cl_device_info paramName) {
	T value;
	OCL_SAFE_CALL(clGetDeviceInfo(device, paramName, sizeof(T), &value, nullptr));
	return value;
}

template<>
std::string GetDeviceInfo(cl_device_id device, cl_device_info paramName) {
	size_t paramSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, paramName, 0, nullptr, &paramSize));

	std::vector<char> deviceInfo(paramSize);
    OCL_SAFE_CALL(clGetDeviceInfo(device, paramName, paramSize, deviceInfo.data(), nullptr));

    return std::string(deviceInfo.data());
}

std::string DeviceTypeToString(cl_device_type deviceType) {
	std::string result;
	
	if (deviceType == CL_DEVICE_TYPE_ALL) {
		return "ALL";
	}
	
	bool first = true;
	auto appendType = [&](cl_device_type flag, const char* name) {
		if (deviceType & flag) {
			if (!first) result += " | ";
			result += name;
			first = false;
		}
	};
	
	appendType(CL_DEVICE_TYPE_CPU, "CPU");
	appendType(CL_DEVICE_TYPE_GPU, "GPU");
	appendType(CL_DEVICE_TYPE_ACCELERATOR, "ACCELERATOR");
	appendType(CL_DEVICE_TYPE_DEFAULT, "DEFAULT");
	appendType(CL_DEVICE_TYPE_CUSTOM, "CUSTOM");
	
	return result.empty() ? "UNKNOWN" : result;
}


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
	cl_uint platformsCount = CL::GetPlatformsCount();
	std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

	// Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
	std::vector<cl_platform_id> platforms = CL::GetPlatformIds(platformsCount);

	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL

		// TODO 1.1
		// Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
		// Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
		// Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
		// Откройте таблицу с кодами ошибок:
		// libs/clew/CL/cl.h:103
		// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
		// Найдите там нужный код ошибки и ее название
		// Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
		// в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
		// Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

		// ANSWER:
		// OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize));
		// CL_INVALID_VALUE is -30
		// CL_INVALID_VALUE if param_name is not one of the supported values or if size in bytes specified by param_value_size is less than size of return type and param_value is not a NULL value.


		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::cout << "    Platform name: " << CL::GetPlatformInfo(platform, CL_PLATFORM_NAME) << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		std::cout << "    Platform vendor: " << CL::GetPlatformInfo(platform, CL_PLATFORM_VENDOR) << std::endl;
		

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")

		std::vector<cl_device_id> devices = CL::GetDeviceIds(platform, CL_DEVICE_TYPE_ALL);
		auto devicesCount = devices.size();

		std::cout << "    Number of OpenCL devices: " << devicesCount << std::endl;


		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			std::cout << "     - Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			cl_device_id device = devices[deviceIndex];
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			std::cout << "       Device name: " << CL::GetDeviceInfo<std::string>(device, CL_DEVICE_NAME) << std::endl;
			std::cout << "       Device vendor: " << CL::GetDeviceInfo<std::string>(device, CL_DEVICE_VENDOR) << std::endl;
			std::cout << "       Device version: " << CL::GetDeviceInfo<std::string>(device, CL_DEVICE_VERSION) << std::endl;
			std::cout << "       Driver version: " << CL::GetDeviceInfo<std::string>(device, CL_DRIVER_VERSION) << std::endl;
			std::cout << "       Device type: " << CL::DeviceTypeToString(CL::GetDeviceInfo<cl_device_type>(device, CL_DEVICE_TYPE)) << std::endl;

			std::cout << std::endl;

			auto toKB = [](cl_ulong bytes) {
				return bytes / 1024;
			};
			auto toMB = [](cl_ulong bytes) {
				return bytes / (1024*1024);
			};
			std::cout << "       Global Memory: "      << toMB(CL::GetDeviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE)) << "MB" << std::endl;
			std::cout << "       Local Memory: "      << toKB(CL::GetDeviceInfo<cl_ulong>(device, CL_DEVICE_LOCAL_MEM_SIZE)) << "KB" << std::endl;
			std::cout << "       Max Alloc Size: "      << toMB(CL::GetDeviceInfo<cl_ulong>(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE)) << "MB" << std::endl;

			std::cout << std::endl;

			std::cout << "       Compute units: " << CL::GetDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS) << std::endl;
			std::cout << "       Max work group size: " << CL::GetDeviceInfo<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE) << std::endl;

		}
	}

	return 0;
}
