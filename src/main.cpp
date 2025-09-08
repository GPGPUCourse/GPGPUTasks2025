#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

const cl_uint BYTES_IN_MEGABYTE = 1024 * 1024;

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
	// libs/clew/CL/cl.h:178
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:178) -> Enter
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

// Helper to get device param info size.
size_t queryDeviceParamSize(cl_device_id device, cl_device_info param_name)
{
	size_t deviceParamSize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, 0, nullptr, &deviceParamSize));

	return deviceParamSize;
};

// Helper to get device param info.
template<typename ParamType>
ParamType queryDeviceParam(cl_device_id device, cl_device_info param_name)
{
	size_t deviceParamSize = queryDeviceParamSize(device, param_name);

	ParamType deviceParam;
	OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, deviceParamSize, &deviceParam, nullptr));

	return deviceParam;
}

// Specialization to process vector creation correctly.
template<>
std::vector<unsigned char> queryDeviceParam(cl_device_id device, cl_device_info param_name)
{
	size_t deviceParamSize = queryDeviceParamSize(device, param_name);

	std::vector<unsigned char> deviceParam(deviceParamSize, 0);
	OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, deviceParamSize, deviceParam.data(), nullptr));

	return deviceParam;
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

	for(cl_uint platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
		// DONE 1.1
		// Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
		// Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
		// Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
		// Откройте таблицу с кодами ошибок:
		// libs/clew/CL/cl.h:178
		// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:178) -> Enter
		// Найдите там нужный код ошибки и ее название
		// Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
		// в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
		// Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.
		//
		// Вывалился код CL_INVALID_VALUE

		// DONE 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr);
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// DONE 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));

		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr);
		std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

		// DONE 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

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

			// Resolve device type name.
			cl_device_type device_type = queryDeviceParam<cl_device_type>(device, CL_DEVICE_TYPE);
			// Exclusive branches are fine, since there is "else" case.
			std::ostringstream device_type_ss;
			if(device_type & CL_DEVICE_TYPE_GPU)
			{
				device_type_ss << "GPU ";
			}
			else if(device_type & CL_DEVICE_TYPE_CPU)
			{
				device_type_ss << "CPU ";
			}
			else
			{
				device_type_ss << "SOMETHING STRANGE";
			}

			std::cout << "        Name: " << queryDeviceParam<std::vector<unsigned char>>(device, CL_DEVICE_NAME).data() << std::endl;
			std::cout << "        Type: " << device_type_ss.str() << std::endl;
			std::cout << "        Mem Size (MB): " << queryDeviceParam<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE) / BYTES_IN_MEGABYTE << std::endl;
			std::cout << "        Mem Cache Size (MB): " << queryDeviceParam<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) / BYTES_IN_MEGABYTE << std::endl;
			std::cout << "        Image Support: " << queryDeviceParam<cl_bool>(device, CL_DEVICE_IMAGE_SUPPORT) << std::endl;
			std::cout << "        OpenCL C version: " << queryDeviceParam<std::vector<unsigned char>>(device, CL_DEVICE_OPENCL_C_VERSION).data() << std::endl;
			std::cout << "        CL Driver version: " << queryDeviceParam<std::vector<unsigned char>>(device, CL_DRIVER_VERSION).data() << std::endl;
		}
	}

	return 0;
}
