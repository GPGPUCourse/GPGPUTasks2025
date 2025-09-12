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

		// OCL_SAFE_CALL(clGetPlatformInfo(platform, int('u'+'w'+'u'), 0, nullptr, &platformNameSize));

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), 0));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));

		std::vector<unsigned char> vendorName(vendorNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize, vendorName.data(), 0));
		std::cout << "    Vendor name: " << vendorName.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;

		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

		std::vector<cl_device_id> devicesId(devicesCount, 0);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devicesId.data(), 0));
		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;

			size_t deviceNameSize = 0;
			cl_device_id deviceId = devicesId[deviceIndex];
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));

			std::vector<unsigned char> deviceName(deviceNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), 0));
			std::cout << "        Device name: " << deviceName.data() << std::endl;

			cl_device_type deviceTypeId = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceTypeId, 0));

			std::string deviceType = "UNRECOGNIZED";
			switch(deviceTypeId)
			{
			case CL_DEVICE_TYPE_CPU:
				deviceType = "CPU";
				break;
			case CL_DEVICE_TYPE_GPU:
				deviceType = "GPU";
			}
			std::cout << "        Device type: " << deviceType << std::endl;

			cl_ulong deviceMemorySizeBytes = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &deviceMemorySizeBytes, 0));
			std::cout << "        Device size(Mb): " << ((double)deviceMemorySizeBytes) / (1024 * 1024) << std::endl;

			cl_bool isLittleEndian = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_ENDIAN_LITTLE, sizeof(cl_bool), &isLittleEndian, 0));
			std::cout << "        Is device little endian: " << (isLittleEndian ? "YES" : "NO") << std::endl;

			cl_bool isImageSupported = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &isImageSupported, 0));
			std::cout << "        Is images supported: " << (isImageSupported ? "YES" : "NO") << std::endl;

			if(isImageSupported)
			{
				size_t max2DImageWidth = 0;
				OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &max2DImageWidth, 0));
				size_t max2DImageHeight = 0;
				OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &max2DImageHeight, 0));
				std::cout << "        Maximum 2D image size(px): " << max2DImageWidth << "x" << max2DImageHeight << std::endl;
				size_t max3dImageDepth = 0;
				OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &max3dImageDepth, 0));
				size_t max3DImageWidth = 0;
				OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &max3DImageWidth, 0));
				size_t max3DImageHeight = 0;
				OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &max3DImageHeight, 0));
				std::cout << "        Maximum 3D image size(px): " << max3DImageWidth << "x" << max3DImageHeight << "x" << max3dImageDepth << std::endl;
			}

			size_t deviceVersionSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_VERSION, 0, nullptr, &deviceVersionSize));

			std::vector<unsigned char> deviceVersion(deviceVersionSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_VERSION, deviceVersionSize, deviceVersion.data(), 0));
			std::cout << "        Device version: " << deviceVersion.data() << std::endl;

			size_t driverVersionSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DRIVER_VERSION, 0, nullptr, &driverVersionSize));

			std::vector<unsigned char> driverVersion(driverVersionSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DRIVER_VERSION, driverVersionSize, driverVersion.data(), 0));
			std::cout << "        Driver version: " << driverVersion.data() << std::endl;
		}
	}

	return 0;
}
