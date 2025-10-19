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

template <typename T>
decltype(auto) get_device_parameter(const cl_device_id& device, const cl_device_info& param_name) {
	T buffer;
	OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, sizeof(T), &buffer, nullptr));
	return buffer;
}

decltype(auto) get_device_parameter(const cl_device_id& device, const cl_device_info& param_name) {
	size_t size = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, 0, nullptr, &size));
	std::vector<unsigned char> buffer(size, 0);
	OCL_SAFE_CALL(clGetDeviceInfo(device, param_name, size, buffer.data(), nullptr));
	return buffer;
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

		// --- SOLUTION ---
			// -30 == CL_INVALID_VALUE
			// clGetPlatformInfo calls CL_INVALID_VALUE if:
			// 	-- param_name is not one of the supported values;
			//  -- if size in bytes specified by param_value_size is less than size of return type and param_value is not a NULL value. 
			// But param_value is nullptr, so the issue is param_name is not one of the supported values.
		// --- SOLUTION END ---

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
		cl_uint return_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount);
		if(return_code == CL_DEVICE_NOT_FOUND) {
			std::cout << "    Number of OpenCL devices: 0" << std::endl;
			continue;
		}
		OCL_SAFE_CALL(return_code);
		std::cout << "    Number of OpenCL devices: " << devicesCount << std::endl;

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
			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			cl_device_id device = devices[deviceIndex];

			// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
			// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL

			std::vector<unsigned char> name = get_device_parameter(device, CL_DEVICE_NAME);
			std::cout << "        Device name: " << name.data() << std::endl;

			cl_device_type type = get_device_parameter<cl_device_type>(device, CL_DEVICE_TYPE);
			std::cout << "        Device type: ";
			if (type & CL_DEVICE_TYPE_CPU) std::cout << "CPU ";
			if (type & CL_DEVICE_TYPE_GPU) std::cout << "GPU ";
			if (type & CL_DEVICE_TYPE_ACCELERATOR) std::cout << "ACCELERATOR ";
			if (type & CL_DEVICE_TYPE_CUSTOM) std::cout << "CUSTOM ";
			if (type & CL_DEVICE_TYPE_DEFAULT) std::cout << "DEFAULT ";
			std::cout << std::endl;

			cl_ulong mem_size = get_device_parameter<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
			std::cout << "        Device global memory size: " << mem_size /1024 /1024 << " Mbytes" << std::endl;

			cl_ulong mem_cache_size = get_device_parameter<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
			std::cout << "        Device global memory cache size: " << mem_cache_size /1024 << " Kbytes" << std::endl;

			cl_uint compute_units = get_device_parameter<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
			std::cout << "        The number of parallel compute units: " << compute_units << std::endl;

			cl_bool is_available = get_device_parameter<cl_bool>(device, CL_DEVICE_AVAILABLE);
			std::cout << "        Device available: " << (is_available ? "true" : "false") << std::endl;

		}
	}

	return 0;
}
