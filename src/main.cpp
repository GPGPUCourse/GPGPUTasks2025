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

		// Поймаем error_code: CL_INVALID_VALUE = -30
		// OCL_SAFE_CALL(clGetPlatformInfo(platform, 52, 0, nullptr, &platformNameSize));


		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformName.size(), platformName.data(), &platformNameSize);
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));
		std::vector<unsigned char> vendorName(vendorNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize, vendorName.data(), &vendorNameSize));
		std::cout << "	Vendor name:" << vendorName.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::vector<cl_device_id> deviceName(devicesCount, 0);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,deviceName.size(),deviceName.data(), nullptr));


		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			cl_device_id device =  deviceName[deviceIndex];

			// - Название устройства
			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
			std::vector<unsigned char> devices(deviceNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, devices.size(), devices.data(), nullptr));
			std::cout << "    Device name: " << devices.data() << std::endl;

			cl_device_type deviceTypeName;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceTypeName, nullptr));

			// - Тип устройства (видеокарта/процессор/что-то странное)
			std::string deviceType = "jopa";
			if (deviceTypeName & CL_DEVICE_TYPE_GPU) deviceType = "GPU";
			else if (deviceTypeName & CL_DEVICE_TYPE_CPU) deviceType = "CPU";
			else if (deviceTypeName & CL_DEVICE_TYPE_ACCELERATOR) deviceType = "Accelerator";
			else if (deviceTypeName & CL_DEVICE_TYPE_DEFAULT) deviceType = "Default";
			else deviceType = "O_O";
			std::cout << "    Device: " << deviceIndex << " type: " << deviceType << std::endl;

			// - Размер памяти устройства в мегабайтах
			cl_ulong deviceSizeName = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &deviceSizeName, nullptr));
			std::cout << "    Global mem size: " << deviceSizeName / (1024*1024) << "MB" << std::endl;
			// --- NVIDIA GeForce RTX 3060 Laptop GPU 6143MB < AMD Accelerated Parallel Processing 6235MB 0_0 WHAT? ---

			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			// --- Узнать версию драйверов ---
			size_t deviceNameVersionDriverSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, nullptr, &deviceNameVersionDriverSize));
			std::vector<unsigned char> devicesVersionDriverName(deviceNameVersionDriverSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DRIVER_VERSION, devicesVersionDriverName.size(), devicesVersionDriverName.data(), nullptr));
			std::cout << "    Device driver version: " << devicesVersionDriverName.data() << std::endl;

			// --- Насколько фуллово OpenCL работает с устройством
			size_t deviceNameProfileSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_PROFILE, 0, nullptr, &deviceNameProfileSize));
			std::vector<unsigned char> devicesNameProfileSizeName(deviceNameProfileSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_PROFILE, devicesNameProfileSizeName.size(), devicesNameProfileSizeName.data(), nullptr));
			std::cout << "    Device profile: " << devicesNameProfileSizeName.data() << std::endl;
		}
	}

	return 0;
}
