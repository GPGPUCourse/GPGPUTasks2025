#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <cstddef>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
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

void reportErrorWithoutException(cl_int err, const std::string &filename, int line)
{
	if(CL_SUCCESS == err)
		return;

	// Таблица с кодами ошибок:
	// libs/clew/CL/cl.h:103
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	std::cout << message;
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)
#define OCL_SAFE_CALL_WITHOUT_EXCEPTION(expr) reportErrorWithoutException(expr, __FILE__, __LINE__)

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
	// TODO 1.1 check
	OCL_SAFE_CALL_WITHOUT_EXCEPTION(clGetPlatformInfo(platforms[0], 2222, 0, nullptr, nullptr));

	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
		// TODO 1.1 check
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

		// TODO 1.2 check
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3 check
		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
		std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

		// Запросите и напечатайте так же в консоль вендора данной платформы

		// TODO 2.1
		//
		std::cout << "________________TASK_2______________\n";
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "count device:" << devicesCount << std::endl;
		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		std::vector<std::pair<std::string, cl_device_info>> data_to_gets = {
			{ "CL_DEVICE_TYPE", CL_DEVICE_TYPE },
			{ "CL_DEVICE_VENDOR_ID", CL_DEVICE_VENDOR_ID },
			{ "CL_DEVICE_MAX_COMPUTE_UNITS", CL_DEVICE_MAX_COMPUTE_UNITS },
			{ "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS", CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS },
			{ "CL_DEVICE_MAX_WORK_GROUP_SIZE", CL_DEVICE_MAX_WORK_GROUP_SIZE },
			{ "CL_DEVICE_MAX_WORK_ITEM_SIZES", CL_DEVICE_MAX_WORK_ITEM_SIZES },
			{ "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR", CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR },
			{ "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT", CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT },
			{ "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT", CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT },
			{ "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG", CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG },
			{ "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT", CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT },
			{ "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE", CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE },
			{ "CL_DEVICE_MAX_CLOCK_FREQUENCY", CL_DEVICE_MAX_CLOCK_FREQUENCY },
			{ "CL_DEVICE_ADDRESS_BITS", CL_DEVICE_ADDRESS_BITS },
			{ "CL_DEVICE_MAX_READ_IMAGE_ARGS", CL_DEVICE_MAX_READ_IMAGE_ARGS },
			{ "CL_DEVICE_MAX_WRITE_IMAGE_ARGS", CL_DEVICE_MAX_WRITE_IMAGE_ARGS },
			{ "CL_DEVICE_MAX_MEM_ALLOC_SIZE", CL_DEVICE_MAX_MEM_ALLOC_SIZE },
			{ "CL_DEVICE_IMAGE2D_MAX_WIDTH", CL_DEVICE_IMAGE2D_MAX_WIDTH },
			{ "CL_DEVICE_IMAGE2D_MAX_HEIGHT", CL_DEVICE_IMAGE2D_MAX_HEIGHT },
			{ "CL_DEVICE_IMAGE3D_MAX_WIDTH", CL_DEVICE_IMAGE3D_MAX_WIDTH },
			{ "CL_DEVICE_IMAGE3D_MAX_HEIGHT", CL_DEVICE_IMAGE3D_MAX_HEIGHT },
			{ "CL_DEVICE_IMAGE3D_MAX_DEPTH", CL_DEVICE_IMAGE3D_MAX_DEPTH },
			{ "CL_DEVICE_IMAGE_SUPPORT", CL_DEVICE_IMAGE_SUPPORT },
			{ "CL_DEVICE_MAX_PARAMETER_SIZE", CL_DEVICE_MAX_PARAMETER_SIZE },
			{ "CL_DEVICE_MAX_SAMPLERS", CL_DEVICE_MAX_SAMPLERS },
			{ "CL_DEVICE_MEM_BASE_ADDR_ALIGN", CL_DEVICE_MEM_BASE_ADDR_ALIGN },
			{ "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE", CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE },
			{ "CL_DEVICE_SINGLE_FP_CONFIG", CL_DEVICE_SINGLE_FP_CONFIG },
			{ "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE", CL_DEVICE_GLOBAL_MEM_CACHE_TYPE },
			{ "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE", CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE },
			{ "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE", CL_DEVICE_GLOBAL_MEM_CACHE_SIZE },
			{ "CL_DEVICE_GLOBAL_MEM_SIZE", CL_DEVICE_GLOBAL_MEM_SIZE },
			{ "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE", CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE },
			{ "CL_DEVICE_MAX_CONSTANT_ARGS", CL_DEVICE_MAX_CONSTANT_ARGS },
			{ "CL_DEVICE_LOCAL_MEM_TYPE", CL_DEVICE_LOCAL_MEM_TYPE },
			{ "CL_DEVICE_LOCAL_MEM_SIZE", CL_DEVICE_LOCAL_MEM_SIZE },
			{ "CL_DEVICE_ERROR_CORRECTION_SUPPORT", CL_DEVICE_ERROR_CORRECTION_SUPPORT },
			{ "CL_DEVICE_PROFILING_TIMER_RESOLUTION", CL_DEVICE_PROFILING_TIMER_RESOLUTION },
			{ "CL_DEVICE_ENDIAN_LITTLE", CL_DEVICE_ENDIAN_LITTLE },
			{ "CL_DEVICE_AVAILABLE", CL_DEVICE_AVAILABLE },
			{ "CL_DEVICE_COMPILER_AVAILABLE", CL_DEVICE_COMPILER_AVAILABLE },
			{ "CL_DEVICE_EXECUTION_CAPABILITIES", CL_DEVICE_EXECUTION_CAPABILITIES },
			{ "CL_DEVICE_QUEUE_PROPERTIES", CL_DEVICE_QUEUE_PROPERTIES },
			{ "CL_DEVICE_NAME", CL_DEVICE_NAME },
			{ "CL_DEVICE_VENDOR", CL_DEVICE_VENDOR },
			{ "CL_DRIVER_VERSION", CL_DRIVER_VERSION },
			{ "CL_DEVICE_PROFILE", CL_DEVICE_PROFILE },
			{ "CL_DEVICE_VERSION", CL_DEVICE_VERSION },
			{ "CL_DEVICE_EXTENSIONS", CL_DEVICE_EXTENSIONS },
			{ "CL_DEVICE_PLATFORM", CL_DEVICE_PLATFORM },
			{ "CL_DEVICE_DOUBLE_FP_CONFIG", CL_DEVICE_DOUBLE_FP_CONFIG },
			{ "CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF", CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF },
			{ "CL_DEVICE_HOST_UNIFIED_MEMORY", CL_DEVICE_HOST_UNIFIED_MEMORY },
			{ "CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR", CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR },
			{ "CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT", CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT },
			{ "CL_DEVICE_NATIVE_VECTOR_WIDTH_INT", CL_DEVICE_NATIVE_VECTOR_WIDTH_INT },
			{ "CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG", CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG },
			{ "CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT", CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT },
			{ "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE", CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE },
			{ "CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF", CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF },
			{ "CL_DEVICE_OPENCL_C_VERSION", CL_DEVICE_OPENCL_C_VERSION },
			{ "CL_DEVICE_LINKER_AVAILABLE", CL_DEVICE_LINKER_AVAILABLE },
			{ "CL_DEVICE_BUILT_IN_KERNELS", CL_DEVICE_BUILT_IN_KERNELS },
			{ "CL_DEVICE_IMAGE_MAX_BUFFER_SIZE", CL_DEVICE_IMAGE_MAX_BUFFER_SIZE },
			{ "CL_DEVICE_IMAGE_MAX_ARRAY_SIZE", CL_DEVICE_IMAGE_MAX_ARRAY_SIZE },
			{ "CL_DEVICE_PARENT_DEVICE", CL_DEVICE_PARENT_DEVICE },
			{ "CL_DEVICE_PARTITION_MAX_SUB_DEVICES", CL_DEVICE_PARTITION_MAX_SUB_DEVICES },
			{ "CL_DEVICE_PARTITION_PROPERTIES", CL_DEVICE_PARTITION_PROPERTIES },
			{ "CL_DEVICE_PARTITION_AFFINITY_DOMAIN", CL_DEVICE_PARTITION_AFFINITY_DOMAIN },
			{ "CL_DEVICE_PARTITION_TYPE", CL_DEVICE_PARTITION_TYPE },
			{ "CL_DEVICE_REFERENCE_COUNT", CL_DEVICE_REFERENCE_COUNT },
			{ "CL_DEVICE_PREFERRED_INTEROP_USER_SYNC", CL_DEVICE_PREFERRED_INTEROP_USER_SYNC },
			{ "CL_DEVICE_PRINTF_BUFFER_SIZE", CL_DEVICE_PRINTF_BUFFER_SIZE },
		};

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			cl_device_id device = devices[deviceIndex];

			for(const std::pair<std ::string, cl_device_info> &data_to_get : data_to_gets)
			{
				size_t size_of_param = 0;
				OCL_SAFE_CALL(clGetDeviceInfo(device, data_to_get.second, 0, nullptr, &size_of_param));
				std::vector<unsigned char> buffer(size_of_param, 0);

				OCL_SAFE_CALL(clGetDeviceInfo(device, data_to_get.second, size_of_param, buffer.data(), nullptr));
				switch(data_to_get.second)
				{
				case CL_DEVICE_NAME:
				case CL_DEVICE_VENDOR:
				case CL_DRIVER_VERSION:
				case CL_DEVICE_PROFILE:
				case CL_DEVICE_VERSION:
				case CL_DEVICE_EXTENSIONS:
				case CL_DEVICE_OPENCL_C_VERSION:
				case CL_DEVICE_BUILT_IN_KERNELS:
					std::cout << data_to_get.first << ": " << buffer.data() << std::endl;
					break;
				case CL_DEVICE_VENDOR_ID:
				case CL_DEVICE_MAX_COMPUTE_UNITS:
				case CL_DEVICE_MAX_READ_IMAGE_ARGS:
				case CL_DEVICE_MAX_WRITE_IMAGE_ARGS:
				case CL_DEVICE_MAX_SAMPLERS:
				case CL_DEVICE_ADDRESS_BITS:
					std::cout << data_to_get.first << ": " << *reinterpret_cast<cl_uint *>(buffer.data()) << std::endl;
					break;
				case CL_DEVICE_GLOBAL_MEM_SIZE:
				case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
				case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:
					std::cout << data_to_get.first << ": " << *reinterpret_cast<cl_ulong *>(buffer.data()) / (1024 * 1024) << " MB" << std::endl;
					break;
				default:
					std::cout << data_to_get.first << "(binary data): size: " << size_of_param << " bytes" << std::endl;
					break;
				}
			}
			// OCL_SAFE_CALL(clGetDeviceInfo())
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
		}
	}

	return 0;
}
