#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <cstddef>
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

void printDeviceTypes(cl_device_type deviceType)
{
	if(deviceType & CL_DEVICE_TYPE_CPU)
	{
		std::cout << "CPU ";
	}
	if(deviceType & CL_DEVICE_TYPE_GPU)
	{
		std::cout << "GPU ";
	}
	if(deviceType & CL_DEVICE_TYPE_ACCELERATOR)
	{
		std::cout << "Accelerator ";
	}
	if(deviceType & CL_DEVICE_TYPE_CUSTOM || deviceType & CL_DEVICE_TYPE_CUSTOM)
	{
		std::cout << "Unknown ";
	}
}

template<typename T>
void getClDeviceInfoParam(cl_device_id device, cl_device_info param_name, T *param_value)
{
	OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, sizeof(T), param_value, nullptr));
}

void getClDeviceInfoParam(cl_device_id device, cl_device_info param_name, std::vector<unsigned char> &buffer)
{
	size_t paramSize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, 0, nullptr, &paramSize));
	buffer.resize(paramSize);

	OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, paramSize, buffer.data(), nullptr));
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
		// Была ошибка CL_INVALID_VALUE

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		// clGetPlatformInfo(...);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0ull, nullptr, &vendorNameSize));

		std::vector<unsigned char> vendorName(vendorNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize, vendorName.data(), nullptr));
		std::cout << "    Vendor name: " << vendorName.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

		std::vector<cl_device_id> devices(devicesCount, 0);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		std::vector<unsigned char> strBuffer;

		std::cout << "\n";

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			auto device = devices[deviceIndex];

			getClDeviceInfoParam(device, CL_DEVICE_NAME, strBuffer);
			std::cout << "        Device name: " << strBuffer.data() << std::endl;

			cl_device_type deviceType = 0;
			getClDeviceInfoParam(device, CL_DEVICE_TYPE, &deviceType);

			std::cout << "        Device type: ";
			printDeviceTypes(deviceType);
			std::cout << std::endl;

			cl_ulong deviceMemorySizeBytes = 0;
			getClDeviceInfoParam(device, CL_DEVICE_GLOBAL_MEM_SIZE, &deviceMemorySizeBytes);
			std::cout << "        Device memory size: "
			          << (deviceMemorySizeBytes / 1024 / 1024) << "MB" << std::endl;

			getClDeviceInfoParam(device, CL_DEVICE_EXTENSIONS, strBuffer);
			std::cout << "        Device extensions: " << strBuffer.data() << std::endl;

			getClDeviceInfoParam(device, CL_DRIVER_VERSION, strBuffer);
			std::cout << "        Device version: " << strBuffer.data() << std::endl;
		}
	}

	return 0;
}
