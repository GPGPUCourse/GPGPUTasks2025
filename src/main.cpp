#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <unordered_map>

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

static const std::unordered_map<cl_device_type, std::string> device_type_map = {{CL_DEVICE_TYPE_CPU, "CPU"},
																				{CL_DEVICE_TYPE_GPU, "GPU"},
																			    {CL_DEVICE_TYPE_ACCELERATOR, "Accelerator"},
																			    {CL_DEVICE_TYPE_DEFAULT, "Default"}, 
																			    {CL_DEVICE_TYPE_CUSTOM, "Custom"}};

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
		// OCL_SAFE_CALL(clGetPlatformInfo(platform, 42, 0, nullptr, &platformNameSize));
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
		size_t actual_size = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), &actual_size));
		std::cout << "    Platform name: " << platformName.data() << std::endl;
		std::cout << "    Actual size: " << actual_size << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		// compute length of CL_PLATFORM_VENDOR
		size_t vendor_name_size = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendor_name_size));

		std::vector<unsigned char> vendor_name(vendor_name_size, 0);
		actual_size = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendor_name_size, vendor_name.data(), &actual_size));
		std::cout << "    Platform vendor: " << vendor_name.data() << std::endl;
		std::cout << "    Actual size: " << actual_size << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Number of devices: " << devicesCount << std::endl;
		
		// allocate array of device ids
		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), &devicesCount));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			
			// get id of current device
			cl_device_id current_device_id = devices[deviceIndex];
			std::cout << "    device number " << deviceIndex << ", device ID: " << current_device_id << std::endl; 
			
			// request len of value size
			size_t value_size = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(current_device_id, CL_DEVICE_NAME, 0, nullptr, &value_size));

			std::vector<unsigned char> value(value_size, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(current_device_id, CL_DEVICE_NAME, value_size, value.data(), &value_size));
			std::cout << "        Device name: " << value.data() << std::endl;
			
			value_size = 0;
			cl_ulong mem_size = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(current_device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, &value_size));
			std::cout << "        Memory size: " << mem_size/1024/1024 << " Mb" << std::endl;

			value_size = 0;
			cl_device_type device_type = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(current_device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, &value_size));
			if (device_type_map.count(device_type))
			{
				std::cout << "        Device type: " << device_type_map.at(device_type) << std::endl;
			}
			else
			{
				std::cout << "        Device type: Unknown" << std::endl;
			}

			value_size = 0;
			value.clear();
			OCL_SAFE_CALL(clGetDeviceInfo(current_device_id, CL_DRIVER_VERSION, 0, nullptr, &value_size));
			value.reserve(value_size);
			OCL_SAFE_CALL(clGetDeviceInfo(current_device_id, CL_DRIVER_VERSION, value_size, value.data(), &value_size));
			std::cout << "        Driver version: " << value.data() << std::endl;

			value_size = 0;
			cl_bool has_compiler = false;
			OCL_SAFE_CALL(clGetDeviceInfo(current_device_id, CL_DEVICE_COMPILER_AVAILABLE, sizeof(cl_bool), &has_compiler, &value_size));
			std::cout << "        Has compiler: " << has_compiler << std::endl;
		}
	}

	return 0;
}
