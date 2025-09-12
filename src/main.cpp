#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

struct Indent
{
	size_t count;

	friend std::ostream &operator<<(std::ostream &os, Indent indent)
	{
		return os << std::setfill(' ') << std::setw(indent.count) << ' ';
	}
};

const Indent TAB = Indent{ 4 };
const Indent DOUBLE_TAB = Indent{ 8 };

void reportError(cl_int err, const std::string &filename, int line)
{
	if(CL_SUCCESS == err)
		return;

	std::ostringstream oss;
	oss << "OpenCL error code " << err << " encountered at " << filename << ":" << line;
	throw std::runtime_error(oss.str());
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

int main()
{
	if(!ocl_init())
		throw std::runtime_error("Can't init OpenCL driver!");

	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
	std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		cl_platform_id platform = platforms[platformIndex];

		auto print_platform_info_string = [=](cl_platform_info param, std::string_view name) {
			size_t paramSize = 0;
			OCL_SAFE_CALL(clGetPlatformInfo(platform, param, 0, nullptr, &paramSize));

			std::vector<unsigned char> paramDescription(paramSize);
			OCL_SAFE_CALL(clGetPlatformInfo(platform, param, paramSize * sizeof(unsigned char), paramDescription.data(), NULL));
			std::cout << TAB << name << ": " << paramDescription.data() << std::endl;
		};

		std::cout << TAB << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;

		print_platform_info_string(CL_PLATFORM_NAME, "Platform name");
		print_platform_info_string(CL_PLATFORM_VENDOR, "Platform vendor");

		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), NULL));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			auto device = devices[deviceIndex];

			auto print_device_info_string = [=](cl_device_info param, std::string_view name) {
				size_t paramSize = 0;
				OCL_SAFE_CALL(clGetDeviceInfo(device, param, 0, nullptr, &paramSize));

				std::vector<unsigned char> paramDescription(paramSize);
				OCL_SAFE_CALL(clGetDeviceInfo(device, param, paramSize * sizeof(unsigned char), paramDescription.data(), NULL));
				std::cout << DOUBLE_TAB << name << ": " << paramDescription.data() << std::endl;
			};

			print_device_info_string(CL_DEVICE_NAME, "Device name");
			print_device_info_string(CL_DEVICE_VENDOR, "Device vendor");

			cl_device_type deviceType = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, NULL));
			std::cout << DOUBLE_TAB << "Device type: " << (deviceType == CL_DEVICE_TYPE_CPU ? "CPU" : deviceType == CL_DEVICE_TYPE_GPU ? "GPU"
			                                                                                                                           : "Other")
			          << std::endl;

			cl_ulong memorySizeInBytes = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memorySizeInBytes, NULL));
			std::cout << DOUBLE_TAB << "Global memory size: " << (memorySizeInBytes / (1024 * 1024)) << "Mb" << std::endl;

			print_device_info_string(CL_DRIVER_VERSION, "Driver version");
			print_device_info_string(CL_DEVICE_VERSION, "Supported OpenCL version");
		}
	}

	return 0;
}
