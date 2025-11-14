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
	std::cerr << "error: " << message << std::endl;
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

template<typename T>
std::vector<T> getDeviceInfoVector(cl_device_id device, cl_device_info info)
{
	size_t size = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, info, 0, nullptr, &size));

	std::vector<T> result(size / sizeof(T), T());
	OCL_SAFE_CALL(clGetDeviceInfo(device, info, size, result.data(), nullptr));

	return result;
}

template<typename T>
T getDeviceInfo(cl_device_id device, cl_device_info info)
{
	return getDeviceInfoVector<T>(device, info).front();
}

std::string getDeviceType(cl_device_type type)
{
	switch(type)
	{
	case CL_DEVICE_TYPE_CPU:
		return "CPU";
	case CL_DEVICE_TYPE_GPU:
		return "GPU";
	case CL_DEVICE_TYPE_ACCELERATOR:
		return "ACCELERATOR";
	case CL_DEVICE_TYPE_DEFAULT:
		return "DEFAULT";
	case CL_DEVICE_TYPE_CUSTOM:
		return "CUSTOM";
	default:
		return "UNKNOWN";
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

		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
		std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			cl_device_id device = devices[deviceIndex];

			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;

			std::vector<unsigned char> deviceName =
			    getDeviceInfoVector<unsigned char>(device, CL_DEVICE_NAME);
			std::cout << "        Device name: " << deviceName.data() << std::endl;

			cl_device_type deviceType =
			    getDeviceInfo<cl_device_type>(device, CL_DEVICE_TYPE);
			std::cout << "        Device type: " << getDeviceType(deviceType) << std::endl;

			constexpr size_t kb = 1024;
			constexpr size_t mb = 1024 * kb;
			constexpr size_t gb = 1024 * mb;

			cl_ulong deviceGlobalMemory =
			    getDeviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
			std::cout << "        Device global memory: " << deviceGlobalMemory / mb << " mb ("
			          << deviceGlobalMemory << " b)" << std::endl;

			cl_ulong deviceGlobalMemoryCacheSize =
			    getDeviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
			std::cout << "        Device global cache size: "
			          << deviceGlobalMemoryCacheSize / kb << " kb" << std::endl;

			cl_uint deviceGlobalMemoryCachelineSize =
			    getDeviceInfo<cl_uint>(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE);
			std::cout << "        Device global cacheline size: "
			          << deviceGlobalMemoryCachelineSize << " b" << std::endl;

			cl_ulong deviceLocalMemorySize =
			    getDeviceInfo<cl_ulong>(device, CL_DEVICE_LOCAL_MEM_SIZE);
			std::cout << "        Device local memory size: "
			          << deviceLocalMemorySize / kb << " kb" << std::endl;

			cl_uint deviceMaxSamplers =
			    getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_SAMPLERS);
			std::cout << "        Device max samplers: " << deviceMaxSamplers
			          << std::endl;

			cl_bool deviceAvailable =
			    getDeviceInfo<cl_bool>(device, CL_DEVICE_AVAILABLE);
			std::cout << "        Device available: "
			          << (deviceAvailable ? "true" : "false") << std::endl;
		}
	}

	return 0;
}
