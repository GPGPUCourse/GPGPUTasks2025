#include "device_info.h"
#include "utils.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cctype>

namespace ocl {

DeviceInfo::DeviceInfo()
{
	device_type					= 0;
	image_support				= false;
	image2d_max_width			= 0;
	image2d_max_height			= 0;
	max_compute_units			= 0;
	max_mem_alloc_size			= 0;
	max_workgroup_size			= 0;
	max_work_item_sizes[0]		= 0;
	max_work_item_sizes[1]		= 0;
	max_work_item_sizes[2]		= 0;
	global_mem_size				= 0;
	device_address_bits			= 0;
	vendor_id					= 0;
	warp_size					= 0;
	wavefront_width				= 0;
	opencl_major_version		= 0;
	opencl_minor_version		= 0;
}

bool DeviceInfo::supportsFreeMemoryRequest() const
{
	return (device_type == CL_DEVICE_TYPE_GPU && vendor_id == gpu::VENDOR::ID_AMD && hasExtension(CL_AMD_DEVICE_ATTRIBUTE_QUERY_EXT));
}

uint64_t DeviceInfo::freeMemory() const
{
	uint64_t free_mem_size;
	if (supportsFreeMemoryRequest()) {
		free_mem_size = 0;

		size_t size = 0;

		cl_int status = CL_SUCCESS;
		OCL_TRACE(status = clGetDeviceInfo(device_id_, CL_DEVICE_GLOBAL_FREE_MEMORY_AMD, 0, NULL, &size));

		if (status != CL_SUCCESS) {
			std::cerr << "Warning: CL_DEVICE_GLOBAL_FREE_MEMORY_AMD has failed (error " << status << ") #1" << std::endl;
		}

		std::vector<unsigned char> buffer(size, 239);

		if (status == CL_SUCCESS) {
			OCL_TRACE(status = clGetDeviceInfo(device_id_, CL_DEVICE_GLOBAL_FREE_MEMORY_AMD, size, buffer.data(), NULL));
		}

		if (status != CL_SUCCESS) {
			std::cerr << "Warning: CL_DEVICE_GLOBAL_FREE_MEMORY_AMD has failed (error " << status << ") #2" << std::endl;
		}

		bool failed = status != CL_SUCCESS;
		bool verbose = false;

		if (!failed) {
			if (size == sizeof(uint32_t) || size == sizeof(uint64_t)) {
				if (size == sizeof(uint32_t)) {
					free_mem_size = *((uint32_t *)buffer.data());
				} else {
					free_mem_size = *((uint64_t *)buffer.data());
				}
				if (free_mem_size > global_mem_size / 1024) {
					if (verbose) {
						std::cerr << "Warning: CL_DEVICE_GLOBAL_FREE_MEMORY_AMD returned too big free memory: ";
					}
					failed = true;
				} else {
					free_mem_size *= 1024; // converting from KBytes to bytes
				}
			} else {
				std::cerr << "Warning: CL_DEVICE_GLOBAL_FREE_MEMORY_AMD returned unexpected size: ";
				failed = true;
			}
		}

		if (failed) {
			if (verbose) {
				std::cerr << "Data[" << size << "]";
				if (size > 0) {
					auto flags = std::cerr.flags();
					std::cerr << "=" << std::hex;
					for (size_t i = 0; i < std::min(size, (size_t) 32); ++i) {
						std::cerr << +(buffer[i]);
						if (i < size - 1) std::cerr << ",";
					}
					std::cerr.flags(flags);
				}
				std::cerr << " free_mem_size=" << free_mem_size << " global_mem_size=" << (global_mem_size / 1024) << std::endl;
			}

			free_mem_size = global_mem_size - global_mem_size / 5;
		}
	} else {
		free_mem_size = global_mem_size - global_mem_size / 5;
	}
	return free_mem_size;
}

void DeviceInfo::print(uint64_t free_mem_size) const
{
	std::cout << "Using device: " << device_name << ", " << max_compute_units << " compute units";
	if (free_mem_size != (uint64_t) -1) {
		std::cout << ", free memory: " << (free_mem_size >> 20) << "/" << (global_mem_size >> 20) << " MB";
	} else {
		std::cout << ", " << (global_mem_size >> 20) << " MB global memory";
	}
	std::cout << ", OpenCL " << opencl_major_version << "." << opencl_minor_version << std::endl;

	std::cout << "  driver version: " << driver_version << ", platform version: " << platform_version << std::endl;
	std::cout << "  max work group size " << max_workgroup_size << std::endl;
	std::cout << "  max work item sizes [" << max_work_item_sizes[0] << ", " << max_work_item_sizes[1] << ", " << max_work_item_sizes[2] << "]" << std::endl;
	std::cout << "  max mem alloc size " << (max_mem_alloc_size >> 20) << " MB" << std::endl;
	if (warp_size != 0)
		std::cout << "  warp size " << warp_size << std::endl;
	if (wavefront_width != 0)
		std::cout << "  wavefront width " << wavefront_width << std::endl;
}

std::string DeviceInfo::getDriverVersion(cl_device_id device_id)
{
	char driver_version_string[1024] = "";
	if (CL_SUCCESS == clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(driver_version_string), &driver_version_string, NULL)) {
		return std::string(driver_version_string);
	} else {
		return "";
	}
}

void DeviceInfo::init(cl_device_id device_id)
{
	device_id_ = device_id;

	cl_device_type	device_type					= 0;
	cl_uint			max_compute_units			= 0;		  // Number of compute units (SM's on NV GPU)
	cl_uint			max_work_item_dimensions	= 0;
	size_t			max_workgroup_size			= 0;
	cl_bool			image_support				= 0;
	size_t			image2d_max_width			= 0;
	size_t			image2d_max_height			= 0;
	cl_uint			vendor_id					= 0;
	cl_ulong		max_mem_alloc_size			= 0;
	cl_ulong		global_mem_size				= 0;
	cl_uint			device_address_bits			= 0;
	char			device_string[1024]			= "";
	char			vendor_string[1024]			= "";
	char			driver_version_string[1024] = "";
	char			platform_version_string[1024]= "";

	cl_platform_id platform_id = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_PLATFORM, sizeof(platform_id), &platform_id, NULL));

	OCL_SAFE_CALL(clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION,				sizeof(platform_version_string),	&platform_version_string, NULL));

	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_NAME,						sizeof(device_string),				&device_string, NULL));
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_VENDOR,						sizeof(vendor_string),				&vendor_string, NULL));
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DRIVER_VERSION,						sizeof(driver_version_string),		&driver_version_string, NULL));
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_TYPE,						sizeof(cl_device_type),				&device_type, NULL));
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,			sizeof(max_compute_units),			&max_compute_units, NULL));
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE,			sizeof(max_mem_alloc_size),			&max_mem_alloc_size, NULL));
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,	sizeof(max_work_item_dimensions),	&max_work_item_dimensions, NULL));
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,			sizeof(max_workgroup_size),			&max_workgroup_size, NULL));
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE,				sizeof(global_mem_size),			&global_mem_size, NULL));
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_ADDRESS_BITS,				sizeof(device_address_bits),		&device_address_bits, NULL));
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_VENDOR_ID,					sizeof(vendor_id),					&vendor_id, NULL));
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT,				sizeof(image_support),				&image_support, NULL));
	if (image_support == CL_TRUE) {
		OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH,		sizeof(image2d_max_width),			&image2d_max_width, NULL));
		OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT,		sizeof(image2d_max_height),			&image2d_max_height, NULL));
	}

	std::vector<size_t> max_work_item_sizes(max_work_item_dimensions);
	OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, max_work_item_dimensions * sizeof(size_t), max_work_item_sizes.data(), NULL));
	for (int i = 0; i < 3; i++)
		this->max_work_item_sizes[i] = max_work_item_sizes[i];

	this->device_name				= std::string(device_string);
	this->vendor_name				= std::string(vendor_string);
	this->device_type				= device_type;
	this->vendor_id					= vendor_id;
	this->image_support				= (image_support == CL_TRUE);
	this->image2d_max_width			= image2d_max_width;
	this->image2d_max_height		= image2d_max_height;
	this->max_compute_units			= max_compute_units;
	this->max_mem_alloc_size		= max_mem_alloc_size;
	this->max_workgroup_size		= max_workgroup_size;
	this->global_mem_size			= global_mem_size;
	this->device_address_bits		= device_address_bits;
	this->max_work_item_dimensions	= max_work_item_dimensions;
	this->driver_version			= std::string(driver_version_string);
	this->platform_version			= std::string(platform_version_string);

	initExtensions(platform_id, device_id);
	initOpenCLVersion(platform_id, device_id);

	if (device_type == CL_DEVICE_TYPE_GPU && vendor_id == gpu::VENDOR::ID_AMD && hasExtension(CL_AMD_DEVICE_ATTRIBUTE_QUERY_EXT)) {
		OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_BOARD_NAME_AMD, sizeof(device_string), &device_string, NULL));
		this->device_name = std::string(device_string) + " (" + this->device_name + ")";
	}

	this->clean_device_name			= cleanDeviceName(this->device_name);

	cl_uint warp_size = 0;
	size_t wavefront_width = 0;
	if (device_type == CL_DEVICE_TYPE_GPU) {
		if (vendor_id == gpu::VENDOR::ID_NVIDIA && hasExtension(CL_NV_DEVICE_ATTRIBUTE_QUERY_EXT)) {
			OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_WARP_SIZE_NV, sizeof(cl_uint), &warp_size, NULL));
		} else if (vendor_id == gpu::VENDOR::ID_AMD && hasExtension(CL_AMD_DEVICE_ATTRIBUTE_QUERY_EXT)) {
			OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_WAVEFRONT_WIDTH_AMD, sizeof(cl_uint), &wavefront_width, NULL));
		}
	}

	this->warp_size			= warp_size;
	this->wavefront_width	= wavefront_width;
}

void DeviceInfo::initOpenCLVersion(cl_platform_id platform_id, cl_device_id device_id)
{
	const int buffer_limit = 1024;
	char buffer[buffer_limit];

	int platform_major_version = 0;
	int platform_minor_version = 0;

	OCL_SAFE_CALL(clGetPlatformInfo (platform_id, CL_PLATFORM_VERSION, buffer_limit, buffer, NULL));
	parseOpenCLVersion(buffer, buffer_limit, platform_major_version, platform_minor_version);

	int device_major_version = 0;
	int device_minor_version = 0;

	OCL_SAFE_CALL(clGetDeviceInfo (device_id, CL_DEVICE_VERSION, buffer_limit, buffer, NULL));
	parseOpenCLVersion(buffer, buffer_limit, device_major_version, device_minor_version);

	if (device_major_version < platform_major_version
		|| (device_major_version == platform_major_version && device_minor_version < platform_minor_version)) {
		opencl_major_version = device_major_version;
		opencl_minor_version = device_minor_version;
	} else {
		opencl_major_version = platform_major_version;
		opencl_minor_version = platform_minor_version;
	}
}

void DeviceInfo::parseOpenCLVersion(char* buffer, int buffer_limit, int& major_version, int& minor_verions)
{
	// For platform:
	// "OpenCL<space><major_version.minor_version><space><platform-specific information>"
	// For device:
	// "OpenCL<space><major_version.minor_version><space><vendor-specific information>"
	int firstSpaceIndex = -1;
	int firstDotIndex = -1;
	int secondSpaceIndex = -1;
	for (int i = 0; i < buffer_limit; i++ ) {
		if (buffer[i] == ' ') {
			if (firstSpaceIndex == -1) {
				firstSpaceIndex = i;
			} else if (secondSpaceIndex == -1) {
				secondSpaceIndex = i;
				buffer[i] = 0;
				break;
			}
		} else if (buffer[i] == '.' && firstDotIndex == -1) {
			firstDotIndex = i;
			buffer[i] = 0;
		}
	}

	major_version = atoi(buffer + firstSpaceIndex + 1);
	minor_verions = atoi(buffer + firstDotIndex + 1);
}

bool DeviceInfo::isAmdGPU() const
{
	return device_type == CL_DEVICE_TYPE_GPU
		   && (vendor_id == gpu::VENDOR::ID_AMD || vendor_name.find("Advanced Micro Devices") != std::string::npos);
}

bool DeviceInfo::isIntelGPU() const
{
	return device_type == CL_DEVICE_TYPE_GPU
		   && (vendor_id == gpu::VENDOR::ID_INTEL || vendor_name.find("Intel") != std::string::npos);
}

void DeviceInfo::initExtensions(cl_platform_id platform_id, cl_device_id device_id)
{
	for (int i = 0; i < 2; ++i) {
		size_t length;
		if (i == 0) {
			OCL_SAFE_CALL(clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, 0, 0, &length));
		} else {
			OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, 0, &length));
		}

		std::vector<char> buffer(length);
		if (i == 0) {
			OCL_SAFE_CALL(clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, sizeof(char) * buffer.size(), buffer.data(), NULL));
		} else {
			OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, sizeof(char) * buffer.size(), buffer.data(), NULL));
		}

		std::string extension = "";
		for (size_t i = 0; i <= buffer.size(); i++) {
			if (i == buffer.size() || buffer[i] == ' ') {
				if (extension.length() > 0) {
					extensions.insert(extension);
					extension = "";
				}
			} else {
				extension += buffer[i];
			}
		}
	}
}

	std::string DeviceInfo::cleanDeviceName(const std::string &raw_name) const
	{
		std::string clean_name(raw_name.size(), 0);
		for (size_t i = 0; i < raw_name.size(); ++i) {
			if (std::isalnum(raw_name[i]) || raw_name[i] == '(' || raw_name[i] == ')' || raw_name[i] == ' ') {
				clean_name[i] = raw_name[i];
			} else {
				clean_name[i] = '_';
			}
		}

		return clean_name;
	}

}
