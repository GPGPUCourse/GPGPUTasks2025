#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

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
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));

		std::vector<unsigned char> platformName(platformNameSize);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize * sizeof(unsigned char), platformName.data(), NULL));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		size_t vendorNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));

		std::vector<unsigned char> vendorName(vendorNameSize);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize * sizeof(unsigned char), vendorName.data(), NULL));
		std::cout << "    Vendor name: " << vendorName.data() << std::endl;

		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), NULL));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));

			std::vector<unsigned char> deviceName(deviceNameSize);
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_NAME, deviceNameSize * sizeof(unsigned char), deviceName.data(), NULL));
			std::cout << "        Device name: " << deviceName.data() << std::endl;

			size_t deviceVendorSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_VENDOR, 0, nullptr, &deviceVendorSize));

			std::vector<unsigned char> deviceVendor(deviceVendorSize);
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_VENDOR, deviceVendorSize * sizeof(unsigned char), deviceVendor.data(), NULL));
			std::cout << "        Device vendor: " << deviceVendor.data() << std::endl;

			cl_device_type deviceType = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, NULL));
			std::cout << "        Device type: " << (deviceType == CL_DEVICE_TYPE_CPU ? "CPU" : deviceType == CL_DEVICE_TYPE_GPU ? "GPU"
			                                                                                                                     : "Other")
			          << std::endl;

			cl_ulong memorySizeInBytes = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memorySizeInBytes, NULL));
			std::cout << "        Global memory size: " << (memorySizeInBytes / (1024 * 1024)) << "Mb" << std::endl;

			size_t deviceDriverVersionSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DRIVER_VERSION, 0, nullptr, &deviceDriverVersionSize));

			std::vector<unsigned char> driverVersion(deviceDriverVersionSize);
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DRIVER_VERSION, deviceDriverVersionSize * sizeof(unsigned char), driverVersion.data(), NULL));
			std::cout << "        Driver version: " << driverVersion.data() << std::endl;

			size_t deviceVersionSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_VERSION, 0, nullptr, &deviceVersionSize));

			std::vector<unsigned char> deviceVersion(deviceVersionSize);
			OCL_SAFE_CALL(clGetDeviceInfo(devices[deviceIndex], CL_DEVICE_VERSION, deviceVersionSize * sizeof(unsigned char), deviceVersion.data(), NULL));
			std::cout << "        Device version: " << deviceVersion.data() << std::endl;
		}
	}

	return 0;
}
