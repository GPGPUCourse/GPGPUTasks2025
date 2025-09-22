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

template<typename Prop, typename Plat>
std::vector<unsigned char> getPlatformProperty(Prop propertyName, Plat platform)
{
	size_t propertySize = 0;
	OCL_SAFE_CALL(clGetPlatformInfo(platform, propertyName, 0, nullptr, &propertySize));
	std::vector<unsigned char> propertyResult(propertySize, 0);
	OCL_SAFE_CALL(clGetPlatformInfo(platform, propertyName, propertySize, propertyResult.data(), nullptr));
	return propertyResult;
}

template<typename T, typename Prop, typename Dev>
std::vector<T> getDeviceProperty(Prop propertyName, Dev device)
{
	size_t propertySize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, propertyName, 0, nullptr, &propertySize));
	std::vector<T> propertyResult(propertySize / sizeof(T), 0);
	OCL_SAFE_CALL(clGetDeviceInfo(device, propertyName, propertySize, propertyResult.data(), nullptr));
	return propertyResult;
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
		//
		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::cout << "    Platform name: " << getPlatformProperty(CL_PLATFORM_NAME, platform).data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		std::cout << "    Platform vendor: " << getPlatformProperty(CL_PLATFORM_VENDOR, platform).data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
		std::cout << "    Number of devices: " << devicesCount << std::endl;

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			std::cout << "        Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			cl_device_id device = devices[deviceIndex];

			std::cout << "        Device Name: " << getDeviceProperty<unsigned char>(CL_DEVICE_NAME, device).data() << std::endl;

			auto deviceType = getDeviceProperty<cl_device_type>(CL_DEVICE_TYPE, device)[0];
			std::string typeName;
			switch(deviceType)
			{
			case CL_DEVICE_TYPE_CPU:
				typeName = "CPU";
				break;
			case CL_DEVICE_TYPE_GPU:
				typeName = "GPU";
				break;
			case CL_DEVICE_TYPE_ACCELERATOR:
				typeName = "ACCELERATOR";
				break;
			case CL_DEVICE_TYPE_CUSTOM:
				typeName = "CUSTOM";
				break;
			case CL_DEVICE_TYPE_DEFAULT:
				typeName = "DEFAULT";
				break;
			default:
				break;
			}
			std::cout << "        Device Type: " << typeName << std::endl;

			cl_ulong memorySizeBytes = getDeviceProperty<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE, device)[0];
			std::cout << "        Memory Size (in Mb): " << (memorySizeBytes / 1024 / 1024) << std::endl;

			cl_ulong cacheSizeBytes = getDeviceProperty<cl_ulong>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, device)[0];
			std::cout << "        Cache Size (in Mb): " << (cacheSizeBytes / 1024 / 1024) << std::endl;

			cl_ulong cacheLineSizeBytes = getDeviceProperty<cl_uint>(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, device)[0];
			std::cout << "        Cache Line Size (in Bytes): " << (cacheLineSizeBytes) << std::endl;
		}
	}

	return 0;
}
