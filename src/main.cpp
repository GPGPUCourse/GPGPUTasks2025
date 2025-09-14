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
		// Никита: попробовал , не понравилось(


		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		// clGetPlatformInfo(...);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformName.size() * sizeof(unsigned char), &platformName[0], NULL));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы

		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));
		std::vector<unsigned char> vendorName(vendorNameSize, 0);
		// clGetPlatformInfo(...);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorName.size() * sizeof(unsigned char), &vendorName[0], NULL));
		std::cout << "    Vendor name: " << vendorName.data() << std::endl;


		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL , 0, nullptr, &devicesCount));
		std::cout << "    Number of OpenCL devices: " << devicesCount << std::endl;

		
		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL , devicesCount, devices.data(), NULL));
		
		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			std::cout << "--------DEVICE #" << deviceIndex+1 << "/" << devicesCount << "--------" <<std::endl; 
			size_t devTypeSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex],CL_DEVICE_NAME , 0, nullptr, &devTypeSize));
			std::vector<unsigned char> devName(devTypeSize, 0);

			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_NAME , devName.size() * sizeof(unsigned char), &devName[0], NULL));
			std::cout << "        DeviceName: " << devName.data() << std::endl;


			cl_device_type devType ;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex],CL_DEVICE_TYPE , sizeof(cl_device_type), &devType, NULL));
			
			switch (devType)
			{
				case CL_DEVICE_TYPE_CPU:
				std::cout << "        DeviceType: CPU"<< std::endl;
				break;

				case CL_DEVICE_TYPE_GPU:
				std::cout << "        DeviceType: GPU"<< std::endl;
				break;

				case CL_DEVICE_TYPE_ACCELERATOR:
				std::cout << "        DeviceType: ACCELERATOR"<< std::endl;
				break;

				case CL_DEVICE_TYPE_CUSTOM:
				std::cout << "        DeviceType: CUSTOM"<< std::endl;
				break;
			}

			cl_ulong GLOBAL_MEM_SIZE ;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex],CL_DEVICE_GLOBAL_MEM_SIZE , sizeof(cl_ulong), &GLOBAL_MEM_SIZE, NULL));
			std::cout << "        Max mem size: "<< GLOBAL_MEM_SIZE/1024/1024 << "Mb" << std::endl;

			cl_uint MAX_CLOCK_FREQUENCY ;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex],CL_DEVICE_MAX_CLOCK_FREQUENCY , sizeof(cl_uint), &MAX_CLOCK_FREQUENCY, NULL));
			std::cout << "        Max clock freq: "<< MAX_CLOCK_FREQUENCY << "MHz" << std::endl;

			cl_uint MAX_COMPUTE_UNITS ;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex],CL_DEVICE_MAX_COMPUTE_UNITS , sizeof(cl_uint), &MAX_COMPUTE_UNITS, NULL));
			std::cout << "        Max compute units: "<< MAX_COMPUTE_UNITS << std::endl;

			size_t MAX_WORK_GROUP_SIZE ;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex],CL_DEVICE_MAX_WORK_GROUP_SIZE , sizeof(size_t), &MAX_WORK_GROUP_SIZE, NULL));
			std::cout << "        Max work group size: "<< MAX_WORK_GROUP_SIZE << std::endl;

			size_t PRINTF_BUFFER_SIZE ;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex],CL_DEVICE_PRINTF_BUFFER_SIZE , sizeof(size_t), &PRINTF_BUFFER_SIZE, NULL));
			std::cout << "        Device printf buffer size  "<< PRINTF_BUFFER_SIZE << "MB" << std::endl;


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
