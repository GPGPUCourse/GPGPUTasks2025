#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cassert>


std::string deviceTypeToString(cl_device_type deviceType)
{
	std::vector<std::string> typesList;
	if (deviceType & CL_DEVICE_TYPE_CPU)
		typesList.push_back("CPU");
	if (deviceType & CL_DEVICE_TYPE_GPU)
		typesList.push_back("GPU");
	if (deviceType & CL_DEVICE_TYPE_ACCELERATOR)
		typesList.push_back("ACCELERATOR");
	if (deviceType & CL_DEVICE_TYPE_DEFAULT)
		typesList.push_back("DEFAULT");

	if (deviceType & CL_DEVICE_TYPE_CUSTOM)
	{
		assert(typesList.size() == 0);
		return "CUSTOM";
	}

	std::string res = typesList[0];
	for (size_t i = 1; i < typesList.size(); ++i) 
	{
		res += " | " + typesList[1];
	}
	return res;	
}


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

		// Ответ:
		// Выпала ошибка: libc++abi: terminating due to uncaught exception of type std::runtime_error: OpenCL error code -30 encountered at /Users/amitin/Desktop/GPGPUTasks2025/src/main.cpp:57
		// cl.h:204 - #define CL_INVALID_VALUE                            -30
		// CL_INVALID_VALUE if param_name is not one of the supported values or if size in bytes specified by param_value_size is less than size of return type and param_value is not a NULL value.

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3 CL_PLATFORM_VENDOR
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));
		std::vector<unsigned char> vendorName(vendorNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize, vendorName.data(), nullptr));
		std::cout << "    Vendor name: " << vendorName.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Number of devices: " << devicesCount << std::endl;
		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства CL_DEVICE_NAME
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			std::cout << "        Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			cl_device_id device = devices[deviceIndex];


			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
			std::vector<unsigned char> deviceName(deviceNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
			std::cout << "        Device name: " << deviceName.data() << std::endl;

			size_t deviceTypeSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, 0, nullptr, &deviceTypeSize));
			cl_device_type deviceType = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, deviceTypeSize, &deviceType, nullptr));
			std::cout << "        Device type: " << deviceTypeToString(deviceType) << std::endl;

			size_t deviceMemorySizeSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, 0, nullptr, &deviceMemorySizeSize));
			cl_ulong deviceMemorySize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, deviceMemorySizeSize, &deviceMemorySize, nullptr));
			std::cout << "        Memory size: " << deviceMemorySize / 1e6 << " mb" << std::endl;

			size_t deviceIsLittleEndianSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, 0, nullptr, &deviceIsLittleEndianSize));
			cl_bool isLittleEndian = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, deviceIsLittleEndianSize, &isLittleEndian, nullptr));
			std::cout << "        Little endian: " << std::boolalpha  << (isLittleEndian == 1) << std::endl;

			size_t deviceDriverVersionSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, nullptr, &deviceDriverVersionSize));
			std::vector<unsigned char> deviceDriverVersion(deviceDriverVersionSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DRIVER_VERSION, deviceDriverVersionSize, deviceDriverVersion.data(), nullptr));
			std::cout << "        Driver version: " << deviceDriverVersion.data() << std::endl;
		}
	}

	return 0;
}
