#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cstring>

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
void fetch_device_param(cl_device_id device, cl_device_info param_name, T& out)
{
	size_t deviceParamSize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, 0, nullptr, &deviceParamSize));

	std::vector<unsigned char> deviceParamBuf(deviceParamSize, 0);
	OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, deviceParamBuf.size(), deviceParamBuf.data(), nullptr));

	if constexpr (std::is_scalar_v<T>)
		std::memcpy(static_cast<void*>(&out), deviceParamBuf.data(), deviceParamBuf.size());
	else // in this case i "hope" that the output is vector type; better type_trait can be used
	{
		out.resize(deviceParamSize);
		std::memcpy(static_cast<void*>(out.data()), deviceParamBuf.data(), deviceParamBuf.size());
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
		const cl_platform_id platform = platforms[platformIndex];

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

		// OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize));

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);

		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformName.size(), platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы

		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));
		std::vector<unsigned char> vendorName(vendorNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorName.size(), vendorName.data(), nullptr));

		std::cout << "    Vendor name: " << vendorName.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = CL_UINT_MAX;
		OCL_SAFE_CALL( clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Number of OpenCL devices for platform #" << (platformIndex + 1) << " : " << devicesCount << std::endl;

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

			const cl_device_id device = devices[deviceIndex];

			{
				std::vector<unsigned char> deviceName;
				fetch_device_param(device, CL_DEVICE_NAME, deviceName);
				std::cout << "        Device name: " << deviceName.data() << std::endl;
			}
			
			{
				cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
				fetch_device_param(device, CL_DEVICE_TYPE, deviceType);

				std::cout << "        Device type: ";

				if (deviceType & CL_DEVICE_TYPE_DEFAULT)
					std::cout << "DEFAULT "; 
				if (deviceType & CL_DEVICE_TYPE_CPU)
					std::cout << "CPU ";
				if (deviceType & CL_DEVICE_TYPE_GPU)
					std::cout << "GPU ";
				if (deviceType & CL_DEVICE_TYPE_ACCELERATOR)
					std::cout << "ACCELERATOR ";
				if (deviceType & CL_DEVICE_TYPE_CUSTOM)
					std::cout << "CUSTOM ";

				std::cout << std::endl;
			}

			{
				cl_ulong deviceGlobalMemory = CL_ULONG_MAX;
				fetch_device_param(device, CL_DEVICE_GLOBAL_MEM_SIZE, deviceGlobalMemory);
				std::cout << "        Global memory size: " << (deviceGlobalMemory >> 20) << " MB" << std::endl;
			}

			{
				cl_bool endianessLittle = CL_FALSE;
				fetch_device_param(device, CL_DEVICE_ENDIAN_LITTLE, endianessLittle);
				std::cout << "        Endianess: " << (endianessLittle ? "Little" : "Big") << std::endl;
			}

			{
				cl_uint maxDeviceFrequency = CL_UINT_MAX;
				fetch_device_param(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, maxDeviceFrequency);
				std::cout << "        Max frequency: " << maxDeviceFrequency << std::endl;
			}
			
			{
				std::vector<unsigned char> deviceExtensions;
				fetch_device_param(device, CL_DEVICE_EXTENSIONS, deviceExtensions);
				std::cout << "        Extensions: " << deviceExtensions.data() << std::endl;
			}
		}
	}

	return 0;
}
