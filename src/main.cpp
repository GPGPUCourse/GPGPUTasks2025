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
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME /*239*/, 0, nullptr, &platformNameSize));
		// DONE 1.1
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
		// ---===---
		// err: -30
		// CL_INVALID_VALUE if param_name is not one of the supported values or if size in bytes specified by param_value_size is less than size of return type and param_value is not a NULL value.
		

		// DONE 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name:     " << platformName.data() << std::endl;

		// DONE 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));

		std::vector<unsigned char> vendorName(vendorNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize, vendorName.data(), nullptr));
		std::cout << "    Vendor name:       " << vendorName.data() << std::endl;

		// DONE 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Number of devices: " << devicesCount << std::endl;

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// DONE 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			cl_device_id device = devices[deviceIndex];

			// Имя
			size_t nameSize = 0;
			clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &nameSize);
			std::vector<char> deviceName(nameSize);
			clGetDeviceInfo(device, CL_DEVICE_NAME, nameSize, deviceName.data(), nullptr);
			std::cout << "        Name:             " << deviceName.data() << std::endl;

			// Тип
			cl_device_type deviceType = 0;
			clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);

			std::string type;
			auto appendText = [](std::string &s, const std::string &suffix) {
				if (!s.empty())
				{
					s += "-";
				}
				s += suffix;
			};
			if (deviceType & CL_DEVICE_TYPE_DEFAULT)
			{
				appendText(type, "DEFAUL");
			}
			if (deviceType & CL_DEVICE_TYPE_CPU)
			{
				appendText(type, "CPU");
			}
			if (deviceType & CL_DEVICE_TYPE_GPU)
			{
				appendText(type, "GPU");
			}
			if (deviceType & CL_DEVICE_TYPE_ACCELERATOR)
			{
				appendText(type, "ACCELERATOR");
			}
			if (deviceType & CL_DEVICE_TYPE_CUSTOM)
			{
				appendText(type, "CUSTOM");
			}

			std::cout << "        Type:             " << type << std::endl;

			// Память
			cl_ulong globalMemorySize = 0;
			clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemorySize), &globalMemorySize, nullptr);
			std::cout << "        Memory:           " << (globalMemorySize / (1024 * 1024)) << " MB" << std::endl;

			// Частота
			cl_uint clockFrequency = 0;
			clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFrequency), &clockFrequency, nullptr);
			(clockFrequency != 0)
			    ? std::cout << "        Frequency:        " << clockFrequency << " MHz" << std::endl
				: std::cout << "        Frequency:        " << "Undefined" << std::endl;

			// Open CL
			size_t deviceVersionSize = 0;
			clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, nullptr, &deviceVersionSize);
			std::vector<char> deviceVersion(deviceVersionSize);
			clGetDeviceInfo(device, CL_DEVICE_VERSION, deviceVersionSize, deviceVersion.data(), nullptr);
			std::cout << "        OpenCL version:   " << deviceVersion.data() << std::endl;

			size_t openclCVersionSize = 0;
			clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, nullptr, &openclCVersionSize);
			std::vector<char> openclCVersion(openclCVersionSize);
			clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, openclCVersionSize, openclCVersion.data(), nullptr);
			std::cout << "        OpenCL C version: " << openclCVersion.data() << std::endl;

			size_t driverVersionSize = 0;
			clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, nullptr, &driverVersionSize);
			std::vector<char> driverVersion(driverVersionSize);
			clGetDeviceInfo(device, CL_DRIVER_VERSION, driverVersionSize, driverVersion.data(), nullptr);
			std::cout << "        Driver version:   " << driverVersion.data() << std::endl;
		}
	}

	return 0;
}
