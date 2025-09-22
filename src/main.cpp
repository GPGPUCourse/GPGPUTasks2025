#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <vector>

#include <string>
#include <iomanip>

#include <sstream>
#include <iostream>
#include <stdexcept>

template<typename T>
std::string to_string(T value) {

	std::ostringstream ss;
	ss << value;

	return ss.str();
}

void report_error(cl_int err, const std::string &filename, int line) {

	if (CL_SUCCESS == err)
	       return;
	
	std::string message = "opencl error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);

	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) report_error(expr, __FILE__, __LINE__)

int main() {
    
    std::cout << "hello opencl" << std::endl;

    if(!ocl_init())
	    throw std::runtime_error("cant init opencl driver");

    cl_uint platforms_count = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platforms_count));

    std::cout << "number of opencl platforms: " << platforms_count << std::endl;

    std::vector<cl_platform_id> platforms(platforms_count);
    OCL_SAFE_CALL(clGetPlatformIDs(platforms_count, platforms.data(), nullptr));

    for(size_t platform_index = 0; platform_index < platforms_count; ++platform_index) {

	    std::cout << "platform #" << (platform_index + 1) << "/" << platforms_count << std::endl;

	    cl_platform_id platform = platforms.at(platform_index);

	    size_t platform_name_size = 0;
	    size_t platform_vendor_size = 0;
	    size_t platform_version_size = 0;
	    size_t platform_extensions_size = 0;

	    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platform_name_size));
	    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platform_vendor_size));
	    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, nullptr, &platform_version_size));
	    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, 0, nullptr, &platform_extensions_size));
	    //OCL_SAFE_CALL(clGetPlatformInfo(platform, 1337, 0, nullptr, &platform_name_size));
	    // throws opencl error code -30 : CL_INVALID_VALUE
	 
	    std::vector<unsigned char> platform_name(platform_name_size, 0), platform_vendor(platform_vendor_size, 0), platform_version(platform_version_size, 0), platform_extensions(platform_extensions_size, 0);

	    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platform_name_size, platform_name.data(), nullptr));
	    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platform_vendor_size, platform_vendor.data(), nullptr));
	    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, platform_version_size, platform_version.data(), nullptr));
	    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, platform_extensions_size, platform_extensions.data(), nullptr));

	    std::cout << "\tplatform name: " << platform_name.data() << "\n\tplatform vendor: " << platform_vendor.data() <<\
		    "\n\tplatform version: " << platform_version.data() << "\n\tplatform extensions: " << platform_extensions.data() << std::endl;

	    cl_uint devices_count = 0;
	    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices_count, nullptr, &devices_count));
	    std::cout << "\n\tnumber of opencl devices: " << devices_count << std::endl;

	    std::vector<cl_device_id> devices(devices_count);
	    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices_count, devices.data(), nullptr));

	    for (size_t device_index = 0; device_index < devices_count; ++device_index) {

		    std::cout << "\t\tdevice #" << (device_index + 1) << "/" << devices_count << std::endl;

		    cl_device_id device = devices.at(device_index);

		    size_t device_name_size = 0;

		    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &device_name_size));

		    std::vector<unsigned char> device_name(device_name_size, 0);
		    cl_device_type device_type;
		    cl_ulong device_memory_size;

		    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, device_name_size, device_name.data(), nullptr));
		    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr));
		    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_memory_size), &device_memory_size, nullptr));

		    // establish device type
		    std::string device_type_str = " ";
		    if (device_type == CL_DEVICE_TYPE_CPU) {

			    device_type_str += "cpu ";
		    } else if (device_type == CL_DEVICE_TYPE_GPU) {

			    device_type_str += "gpu ";
		    } else if (device_type == CL_DEVICE_TYPE_ACCELERATOR) {

			    device_type_str += "accelerator ";
		    } else if (device_type == CL_DEVICE_TYPE_DEFAULT) {

			    device_type_str += "default ";
		    } else if (device_type == CL_DEVICE_TYPE_CUSTOM) {

			    device_type_str += "custom ";
		    }

		    device_type_str.pop_back();

		    std::cout << "\t\t\tdevice name: " << device_name.data() << "\n\t\t\tdevice type:" << device_type_str <<\
			    "\n\t\t\tdevice memory size: " << std::fixed << static_cast<cl_ulong>(device_memory_size / (1024.0f * 1024.0f)) << " mbytes" << std::endl;

		    // xtra
		    // last lecture touched on work groups
		    size_t device_max_work_group_size = 0;

		    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(device_max_work_group_size), &device_max_work_group_size, nullptr));

		    std::cout << "\t\t\tdevice max work group size: " << device_max_work_group_size << std::endl; 

		    // also had intro to nsight profiler
		    size_t device_profiling_timer_resolution = 0;

		    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(device_profiling_timer_resolution), &device_profiling_timer_resolution, nullptr));

		    std::cout << "\t\t\tdevice profiling timer resolution: " << device_profiling_timer_resolution << " ns" << std::endl; 

		    // vendor why not
		    size_t device_vendor_size = 0;
		    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, nullptr, &device_vendor_size));
		    
		    std::vector<unsigned char> device_vendor(device_vendor_size, 0);

		    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VENDOR, device_vendor_size, device_vendor.data(), nullptr));
		    
		    std::cout << "\t\t\tdevice vendor: " << device_vendor.data() << std::endl; 
	    }
    }

    return 0;
}
