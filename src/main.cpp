#ifdef _WIN32
#include "CL/cl_d3d10.h"
#endif

#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <cmath>
#include <cstdio>
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

	for (cl_platform_id x: platforms)
	{
		std::cout << x<< std::endl;
	}

	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		size_t platformNameSize = 0;
		std::cout << CL_PLATFORM_NAME;
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
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), &platformNameSize);
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));

		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), &platformVendorSize);
		std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;


		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;

		OCL_SAFE_CALL(clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, 100, nullptr, &devicesCount ));
		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, 100, devices.data(), &devicesCount ));
		// for (cl_device_id y: devices)
		// {
		// 	// std::cout << y <<  ' ' << devices[0] << std::endl;
		// }
		std::cout << "    Number of devices: " << devicesCount << std::endl;
		for(cl_device_id deviceIndex: devices)
		{
			// TODO 2.2
			std::cout << "Device #" << deviceIndex << "/" << devicesCount << std::endl;

			// Запросите и напечатайте в консоль:
			// - Название устройства

			size_t deviceNamesize = 0;
			OCL_SAFE_CALL( clGetDeviceInfo(deviceIndex, CL_DEVICE_NAME, 0, nullptr, &deviceNamesize ));
			std::vector<unsigned char> deviceName(deviceNamesize, 0);
			OCL_SAFE_CALL( clGetDeviceInfo(deviceIndex, CL_DEVICE_NAME, deviceNamesize, deviceName.data(), &deviceNamesize ));
			std::cout << "    Device name: " << deviceName.data() << std::endl;

			// - Тип устройства (видеокарта/процессор/что-то странное)

			// size_t deviceTypeSize = 0;
			// OCL_SAFE_CALL( clGetDeviceInfo(deviceIndex, CL_DEVICE_TYPE, 0, nullptr, &deviceTypeSize ));
			// std::vector<unsigned char> deviceType(deviceTypeSize, 0);
			// OCL_SAFE_CALL( clGetDeviceInfo(deviceIndex, CL_DEVICE_TYPE, deviceTypeSize, deviceType.data(), &deviceTypeSize ));
			// std::cout << "    Device type: " << deviceType.data() << std::endl;

			cl_device_type type = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceIndex,
										  CL_DEVICE_TYPE,
										  sizeof(cl_device_type),
										  &type,
										  nullptr));

			std::string typeStr;
			if (type & CL_DEVICE_TYPE_CPU)        typeStr = "CPU";
			else if (type & CL_DEVICE_TYPE_GPU)   typeStr = "GPU";
			else if (type & CL_DEVICE_TYPE_ACCELERATOR) typeStr = "Accelerator";
			else if (type & CL_DEVICE_TYPE_DEFAULT)     typeStr = "Default";
			else typeStr = "Unknown";

			std::cout << "    Device type: " << typeStr << std::endl;


			// - Размер памяти устройства в мегабайтах
			// size_t deviceMemSize = 0;
			// OCL_SAFE_CALL( clGetDeviceInfo(deviceIndex, CL_DEVICE_GLOBAL_MEM_SIZE, 0, nullptr, &deviceMemSize ));
			// std::cout << "    Device mem: " << deviceMemSize << std::endl;

			cl_ulong deviceMem = 0;
			OCL_SAFE_CALL( clGetDeviceInfo(deviceIndex, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &deviceMem, nullptr ));
			std::cout << "    Device mem: " << deviceMem / std::pow(1024, 3) << std::endl;
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересным
			//
			//
			cl_bool  deviceECC = FALSE;
			OCL_SAFE_CALL( clGetDeviceInfo(deviceIndex, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(cl_bool ), &deviceECC, nullptr ));
			std::cout << "    Device ECC: " << deviceECC << std::endl;


			size_t deviceExtension = 0;
			OCL_SAFE_CALL( clGetDeviceInfo(deviceIndex, CL_DEVICE_EXTENSIONS, 0, nullptr, &deviceExtension ));
			std::vector<unsigned char> deviceExt(deviceExtension, 0);
			OCL_SAFE_CALL( clGetDeviceInfo(deviceIndex, CL_DEVICE_EXTENSIONS, deviceExtension, deviceExt.data(), &deviceExtension ));
			std::cout << "    Device extension: " << deviceExt.data() << std::endl;
		}
	}

	return 0;
}
