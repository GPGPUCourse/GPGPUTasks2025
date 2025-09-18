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

std::vector<unsigned char> getDeviceInfoValue(cl_device_id device, cl_device_info info) {
	size_t deviceValueSize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, info, 0,
		nullptr, &deviceValueSize));

	std::vector<unsigned char> deviceValue(deviceValueSize, 0);
	OCL_SAFE_CALL(clGetDeviceInfo(device, info, deviceValueSize,
		deviceValue.data(), nullptr));

	return deviceValue;
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
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr,
			&platformNameSize));

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

		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(),
			nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr,
			&vendorNameSize));

		std::vector<unsigned char> vendorName(vendorNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize,
			vendorName.data(), nullptr));
		std::cout << "    Vendor: " << vendorName.data() << std::endl;

		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount,
			devices.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			cl_device_id device = devices[deviceIndex];

			std::cout << "        Device name: "
				<< getDeviceInfoValue(device, CL_DEVICE_NAME).data() << std::endl;

			std::vector<unsigned char> device_type_data = getDeviceInfoValue(device, CL_DEVICE_TYPE);
			cl_device_type device_type = (device_type_data.size() >= 8 ?
				*reinterpret_cast<cl_device_type *>(device_type_data.data()) : CL_DEVICE_TYPE_CUSTOM);

			std::string device_type_str;
			switch(device_type) {
				case CL_DEVICE_TYPE_CPU:
					device_type_str = "CPU";
					break;
				case CL_DEVICE_TYPE_GPU:
					device_type_str = "GPU";
					break;
				case CL_DEVICE_TYPE_ACCELERATOR:
					device_type_str = "ACCELERATOR";
					break;
				case CL_DEVICE_TYPE_DEFAULT:
					device_type_str = "DEFAULT";
					break;
				default:
					device_type_str = "UNKNOWN";
					break;
			}
			std::cout << "        Device type: " << device_type_str << std::endl;

			std::vector<unsigned char> device_mem_size_data =
				getDeviceInfoValue(device, CL_DEVICE_GLOBAL_MEM_SIZE);
			cl_ulong mem_size = (device_mem_size_data.size() >= 8 ?
				*reinterpret_cast<cl_ulong *>(device_mem_size_data.data()) : 0);
			std::cout << "        Memory size: " << mem_size / 1024 / 1024
				<< " MB" << std::endl;

			std::vector<unsigned char> device_freq_data =
				getDeviceInfoValue(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
			cl_uint freq = (device_freq_data.size() >= 4 ?
				*reinterpret_cast<cl_uint *>(device_freq_data.data()) : 0);
			std::cout << "        Clock frequency: " << freq << " MHz" << std::endl;

			std::vector<unsigned char> device_version_data = getDeviceInfoValue(device, CL_DEVICE_VERSION);
			std::cout << "        Device version: " << device_version_data.data() << std::endl;


			std::vector<unsigned char> device_driver_data = getDeviceInfoValue(device, CL_DRIVER_VERSION);
			std::cout << "        Driver version: " << device_driver_data.data() << std::endl;
		}
	}

	return 0;
}
