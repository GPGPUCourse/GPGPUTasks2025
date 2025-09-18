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
	std::cout << message << std::endl;  // на msvc не показывается текст исключения
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
		// OCL_SAFE_CALL(clGetPlatformInfo(platform, 3284, 0, nullptr, &platformNameSize));  
		// -30 - CL_INVALID_VALUE - неподдерживаемый параметр, или буфера не хватит для записи значения параметра (когда просим записать)
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
		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
		std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Number of devices: " << devicesCount << std::endl;

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			auto deviceID = devices[deviceIndex];
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceID, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
			std::vector<unsigned char> deviceName(deviceNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(deviceID, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
			std::cout << "        Device name: " << deviceName.data() << std::endl;
			// не знаю, почему ждля видеокарты не пишет нормальное имя (rx 6600m), драйвера свежие

			cl_device_type deviceType = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceID, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
			std::string deviceTypeStr;
			if(deviceType & CL_DEVICE_TYPE_CPU)
				deviceTypeStr += "CPU ";
			if(deviceType & CL_DEVICE_TYPE_GPU)
				deviceTypeStr += "GPU ";
			if(deviceType & CL_DEVICE_TYPE_ACCELERATOR)
				deviceTypeStr += "ACCELERATOR ";
			if(deviceType & CL_DEVICE_TYPE_DEFAULT)
				deviceTypeStr += "DEFAULT ";
			if(deviceType & CL_DEVICE_TYPE_CUSTOM)
				deviceTypeStr += "CUSTOM ";
			std::cout << "            Device type: " << deviceTypeStr << std::endl;

			cl_ulong deviceMemSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceMemSize), &deviceMemSize, nullptr));
			std::cout << "            Device memory size: " << (deviceMemSize / (1024 * 1024)) << " MB" << std::endl;

			cl_ulong deviceCacheSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(deviceCacheSize), &deviceCacheSize, nullptr));
			std::cout << "            Device cache size: " << (deviceCacheSize / 1024) << " KB" << std::endl;

			cl_uint deviceCUcnt = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(deviceCUcnt), &deviceCUcnt, nullptr));
			std::cout << "            Device compute units: " << deviceCUcnt << std::endl;

			size_t deviceVersionLength = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceID, CL_DEVICE_VERSION, 0, nullptr, &deviceVersionLength));
			std::vector<unsigned char> deviceVersion(deviceVersionLength, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(deviceID, CL_DEVICE_VERSION, deviceVersionLength, deviceVersion.data(), nullptr));
			std::cout << "            Device version: " << deviceVersion.data() << std::endl;

			size_t driverVersionLength = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceID, CL_DRIVER_VERSION, 0, nullptr, &driverVersionLength));
			std::vector<unsigned char> driverVersion(driverVersionLength, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(deviceID, CL_DRIVER_VERSION, driverVersionLength, driverVersion.data(), nullptr));
			std::cout << "            Driver version: " << driverVersion.data() << std::endl;
		}
	}

	return 0;
}
