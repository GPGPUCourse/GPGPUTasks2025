#include "CL/cl_platform.h"
#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <cerrno>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
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

template <typename T>
T getDeviceInfoByParam(cl_device_id id, cl_device_info info) {
	size_t size = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(id, info, 0, nullptr, &size));
	std::vector<char> data(size);
	OCL_SAFE_CALL(clGetDeviceInfo(id, info, size, data.data(), nullptr));
	if constexpr (std::is_same_v<T, std::string>) {
		return std::string(data.data(), data.size());
	} else {
		return *static_cast<T*>(static_cast<void*>(data.data()));
	}
}

std::string getDiviceTypeName(cl_device_type type) {
	switch (type) {
		case CL_DEVICE_TYPE_CPU:
			return "CPU";
		case CL_DEVICE_TYPE_GPU:
			return "GPU";
		case CL_DEVICE_TYPE_ACCELERATOR:
			return "ACCELERATOR";
		case CL_DEVICE_TYPE_DEFAULT:
			return "DEFAULT"; // whatevet it means
		default:
			return "CUSTOM";
	}
	return "UNREACHABLE";
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

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
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

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
		std::vector<unsigned char> platformVendorName(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendorName.data(), nullptr));
		std::cout << "    Platform Vendor: " << platformVendorName.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

		std::vector<cl_device_id> deviceIds(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, deviceIds.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			const auto id = deviceIds[deviceIndex];
			std::string padding = std::string(8, ' ');
			std::cout << padding << "- Name: " << getDeviceInfoByParam<std::string>(id, CL_DEVICE_NAME) << '\n';
			std::cout << padding << "- Type: " << getDiviceTypeName(getDeviceInfoByParam<cl_device_type>(id, CL_DEVICE_TYPE)) << '\n';
			std::cout << padding << "- TotalMemory(Mb): " << getDeviceInfoByParam<cl_ulong>(id, CL_DEVICE_GLOBAL_MEM_SIZE) / 1024 / 1024 << '\n';
			std::cout << padding << "- ImageSupport: " << getDeviceInfoByParam<cl_bool>(id, CL_DEVICE_IMAGE_SUPPORT) << '\n';
			std::cout << padding << "- Profile: " << getDeviceInfoByParam<std::string>(id, CL_DEVICE_PROFILE) << '\n';
			std::cout << padding << "- LocalMem(Kb): " << getDeviceInfoByParam<cl_ulong>(id, CL_DEVICE_LOCAL_MEM_SIZE) / 1024 << '\n';
		}
	}

	return 0;
}
