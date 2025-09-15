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
	cl_uint platformsCount = 30239;
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
		// DONE: Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
		// Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
		// Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
		// Откройте таблицу с кодами ошибок:
		// libs/clew/CL/cl.h:103
		// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
		// Найдите там нужный код ошибки и ее название 
		// #define CL_INVALID_VALUE                            -30
		// CL_INVALID_VALUE if param_name is not one of the supported values or if size in bytes specified by param_value_size is less than size of return type and param_value is not a NULL value.
		// Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
		// в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
		// Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), &platformNameSize));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
		std::vector<unsigned char> platformVendorName(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendorName.data(), &platformVendorSize));
		std::cout << "    Vendor name   : " << platformVendorName.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 228, nullptr, &devicesCount));
		std::cout << "    Device count   : " << devicesCount << std::endl;

		std::vector<cl_device_id> deviceName(devicesCount, 0);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 228, deviceName.data(), &devicesCount));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			cl_device_id deviceId = deviceName[deviceIndex];
			std::cout << "    	Device id     : " << deviceId << std::endl;

			// name
			size_t nameLen = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 0, nullptr, &nameLen));
			std::vector<unsigned char> deviceName(nameLen, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, nameLen, deviceName.data(), &nameLen));
			std::cout << "    	Device name     : " << deviceName.data() << std::endl;

			// type
			size_t typeLen = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, 0, nullptr, &typeLen));
			cl_device_type deviceType;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, typeLen, &deviceType, &typeLen));
			std::cout << "    	Device type     : " << deviceType << std::endl;
			
			// mem
			size_t memLen = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, 0, nullptr, &memLen));
			cl_ulong deviceMem;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, memLen, &deviceMem, &memLen));
			std::cout << "    	Device mem     : " << deviceMem / 1024 / 1024 << " Mb" << std::endl;
			
			// version
			size_t versionLen = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_VERSION, 0, nullptr, &versionLen));
			std::vector<unsigned char> deviceVersion(versionLen, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_VERSION, versionLen, deviceVersion.data(), &versionLen));
			std::cout << "    	Device version     : " << deviceVersion.data() << std::endl;
			
			// driver
			size_t driverLen = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DRIVER_VERSION, 0, nullptr, &driverLen));
			std::vector<unsigned char> driverVersion(driverLen, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DRIVER_VERSION, driverLen, driverVersion.data(), &driverLen));
			std::cout << "    	Driver version     : " << driverVersion.data() << std::endl;

			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
		}
	}

	return 0;
}
