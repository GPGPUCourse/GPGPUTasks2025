#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#define OFFSET1 "    "
#define OFFSET2 OFFSET1 "    "

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

cl_int reportErrorsExceptInvalidValue(cl_int err, const std::string &filename, int line)
{
	if(CL_SUCCESS == err || CL_INVALID_VALUE == err)
		return err;
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SEMI_SAFE_CALL(expr) reportErrorsExceptInvalidValue(expr, __FILE__, __LINE__)

std::string getPlatformData(cl_platform_id platform, cl_platform_info param_name, size_t size = 256)
{
	std::string data(size, 0);
	size_t real_data_size = 0;
	auto res = OCL_SEMI_SAFE_CALL(clGetPlatformInfo(platform, param_name, data.size(), data.data(), &real_data_size));
	if(res == CL_INVALID_VALUE && real_data_size > data.size())
	{
		return getPlatformData(platform, param_name, real_data_size);
	}
	data.resize(real_data_size);
	return data;
}

std::string getDeviceData(cl_device_id device_id, cl_device_info param_name, size_t size = 256)
{
	std::string data(size, 0);
	size_t real_data_size = 0;
	auto res = OCL_SEMI_SAFE_CALL(clGetDeviceInfo(device_id, param_name, data.size(), data.data(), &real_data_size));
	if(res == CL_INVALID_VALUE && real_data_size > data.size())
	{
		return getDeviceData(device_id, param_name, real_data_size);
	}
	data.resize(real_data_size);
	return data;
}

std::string getDeviceType(cl_device_id device_id)
{
	cl_device_type type;
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, nullptr));
	switch(type)
	{
	case CL_DEVICE_TYPE_CPU:
		return "CPU";
	case CL_DEVICE_TYPE_GPU: return "GPU";
	case CL_DEVICE_TYPE_ACCELERATOR: return "Accelerator";
	case CL_DEVICE_TYPE_CUSTOM: return "Custom";
	default:
		return "Unknown";
	}
}

template<typename T>
std::pair<T, T> div(T x, T y)
{
	auto res = std::div(x, y);
	return { res.quot, res.rem };
}

std::string getDeviceLocalMemorySize(cl_device_id device_id)
{
	cl_ulong memory;
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &memory, nullptr));
	for(auto suff : { "B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB" })
	{
		if(auto [next, rem] = div(memory, 1024); next > 0 && rem == 0)
		{
			memory = next;
		}
		else
		{
			return std::to_string(memory) + suff;
		}
	}
	return "undefined";
}

std::string getDeviceGlobalMemorySize(cl_device_id device_id)
{
	cl_ulong memory;
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memory, nullptr));
	return std::to_string(memory >> 20) + "MB";
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
		std::cout << OFFSET1 "Platform name: " << getPlatformData(platform, CL_PLATFORM_NAME) << "\n";
		std::cout << OFFSET1 "Vendor name:   " << getPlatformData(platform, CL_PLATFORM_VENDOR) << "\n";

		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::vector devices(devicesCount, cl_device_id{});
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices.size(), devices.data(), 0));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			std::cout << OFFSET1 "Device #" << (deviceIndex + 1) << "/" << devicesCount << "\n";
			std::cout << OFFSET2 "Device name:      " << getDeviceData(devices[deviceIndex], CL_DEVICE_NAME) << "\n";
			std::cout << OFFSET2 "Device type:      " << getDeviceType(devices[deviceIndex]) << "\n";
			std::cout << OFFSET2 "Global memory sz: " << getDeviceGlobalMemorySize(devices[deviceIndex]) << "\n";
			std::cout << OFFSET2 "Local memory sz:  " << getDeviceLocalMemorySize(devices[deviceIndex]) << "\n";
			std::cout << OFFSET2 "Device version:   " << getDeviceData(devices[deviceIndex], CL_DEVICE_VERSION) << "\n";
			std::cout << OFFSET2 "Extensions:       " << getDeviceData(devices[deviceIndex], CL_DEVICE_EXTENSIONS) << "\n";
		}
		std::cout.flush();
	}

	return 0;
}
