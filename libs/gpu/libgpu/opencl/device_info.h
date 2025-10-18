#pragma once

#include <cstddef>
#include <string>
#include <set>
#include <cstdint>

typedef struct _cl_platform_id *    cl_platform_id;
typedef struct _cl_device_id *      cl_device_id;

namespace ocl {

class DeviceInfo {
public:
	DeviceInfo();

	void init(cl_device_id device_id);
	void print(uint64_t free_mem_size = (uint64_t) -1) const;

	bool				supportsFreeMemoryRequest() const;
	uint64_t			freeMemory() const;

	static std::string	getDriverVersion(cl_device_id device_id);

	bool				isAmdGPU() const;
	bool 				isIntelGPU() const;
	bool				hasExtension(std::string extension) const	{ return extensions.count(extension) > 0; }

	std::pair<int, int>	openclVersion() const {
		return std::make_pair(opencl_major_version, opencl_minor_version);
	}

	std::string				device_name;
	std::string				clean_device_name;
	std::string				vendor_name;
	uint64_t				device_type;
	uint32_t				vendor_id;
	bool					image_support;
	size_t					image2d_max_width;  // w.r.t. specification: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html
	size_t					image2d_max_height; // if (image_support) then (max_width and max_height >= 8192)
	size_t					max_compute_units;
	uint64_t				max_mem_alloc_size;
	size_t					max_workgroup_size;
	size_t					max_work_item_sizes[3];
	uint64_t				global_mem_size;
	size_t 					device_address_bits;
	size_t					max_work_item_dimensions;
	uint32_t				warp_size;
	size_t					wavefront_width;
	std::string				driver_version;
	std::string				platform_version;

	int 					opencl_major_version;
	int 					opencl_minor_version;

	std::set<std::string>	extensions;

protected:
	void				initExtensions(cl_platform_id platform_id, cl_device_id device_id);
	void				initOpenCLVersion(cl_platform_id platform_id, cl_device_id device_id);
	void				parseOpenCLVersion(char* buffer, int buffer_limit, int& major_version, int& minor_verions);
	std::string			cleanDeviceName(const std::string &device_name) const;

	cl_device_id		device_id_;
};

}
