#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
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

        // try {
        //     OCL_SAFE_CALL(clGetPlatformInfo(platform, 42, 0, nullptr, &platformNameSize));
        // } catch (const std::runtime_error& exception) {
        //     std::cout << exception.what() << std::endl;
        //     return 0;
        // }
        // Output: OpenCL error code -30 encountered at <path_to_repo>/src/main.cpp:71

        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::string platformName;
        platformName.reserve(platformNameSize);

		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
        size_t platformVendorSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));

        std::string platformVendor;
        platformVendor.reserve(platformVendorSize);

		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
		std::cout << "    Vendor name: " << platformVendor.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

        std::vector<cl_device_id> devices;
        devices.reserve(devicesCount);

        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        std::cout << std::endl;
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

            size_t deviceNameSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));

            std::string deviceName;
            deviceName.reserve(deviceNameSize);

            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
            std::cout << "        Device name: " << deviceName.data() << std::endl;

            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));

            if ((deviceType & (deviceType - 1)) != 0) {
                throw std::runtime_error("Invalid device type!");
            }

            if (deviceType == CL_DEVICE_TYPE_DEFAULT) {
                std::cout << "        Device type: default" << std::endl;
            } else if (deviceType == CL_DEVICE_TYPE_CPU) {
                std::cout << "        Device type: cpu" << std::endl;
            } else if (deviceType == CL_DEVICE_TYPE_GPU) {
                std::cout << "        Device type: gpu" << std::endl;
            } else if (deviceType == CL_DEVICE_TYPE_ACCELERATOR) {
                std::cout << "        Device type: accelerator" << std::endl;
            } else if (deviceType == CL_DEVICE_TYPE_CUSTOM) {
                std::cout << "        Device type: custom" << std::endl;
            } else {
                std::cout << "        Device type: unknown" << std::endl;
            }

            cl_ulong deviceMemory;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &deviceMemory, nullptr));
            std::cout << "        Device memory size: " << static_cast<double>(deviceMemory) / (1024 * 1024) << "MB" << std::endl;

            size_t deviceVersionSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, nullptr, &deviceVersionSize));

            std::string deviceVersion;
            deviceVersion.reserve(deviceVersionSize);

            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, deviceVersionSize, deviceVersion.data(), nullptr));
            std::cout << "        Device version: " << deviceVersion.data() << std::endl;

            size_t driverVersionSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, nullptr, &driverVersionSize));

            std::string driverVersion;
            driverVersion.reserve(driverVersionSize);

            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DRIVER_VERSION, driverVersionSize, driverVersion.data(), nullptr));
            std::cout << "        Driver version: " << driverVersion.data() << std::endl;
		}
	}

	return 0;
}


