#define CL_TARGET_OPENCL_VERSION 210
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <libgpu/opencl/utils.h>
#include <libgpu/opencl/device_info.h>
#include <libclew/ocl_init.h>
#include <libgpu/device.h>
#include "libbase/string_utils.h"
#include "libbase/timer.h"
#include <algorithm>
#include <CL/cl.h>
#include <iostream>
#include "enum.h"

#define OCL_CPU_DEVICES_ENABLED true

bool OpenCLEnum::printInfo(cl_device_id id)
{
	timer tm;

	ocl::DeviceInfo info;
	info.init(id);

	double tm_properties = tm.elapsed();
	tm.restart();

	uint64_t free_mem_size = (uint64_t) -1;
	if (info.supportsFreeMemoryRequest())
		free_mem_size = info.freeMemory();

	double tm_memory = tm.elapsed();

	info.print(free_mem_size);

	double tm_total = tm_properties + tm_memory;
	if (tm_total > 0.5)
		std::cout << "  got device properties in " << tm_properties << " sec, free memory in " << tm_memory << " sec" << std::endl;

	return true;
}

OpenCLEnum::OpenCLEnum()
{
}

OpenCLEnum::~OpenCLEnum()
{
}

bool OpenCLEnum::enumPlatforms(bool silent)
{
	cl_uint num_platforms; 
	cl_int ciErrNum;

	// Get OpenCL platform count
	OCL_TRACE(ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms));
	if (ciErrNum != CL_SUCCESS) {
		if (!silent) std::cerr << "clGetPlatformIDs failed: " << ocl::errorString(ciErrNum) << std::endl;
		return false;
	}

	if (num_platforms == 0)
		return true;

	std::vector<cl_platform_id> clPlatformIDs(num_platforms);

	// get platform info for each platform and trap the NVIDIA platform if found
	OCL_TRACE(ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs.data(), NULL));
	if (ciErrNum != CL_SUCCESS) {
		if (!silent) std::cerr << "clGetPlatformIDs for " << num_platforms << " platforms failed: " << ocl::errorString(ciErrNum) << std::endl;
		return false;
	}

	for (cl_uint i = 0; i < num_platforms; ++i) {
		Platform platform;

		cl_platform_id platform_id = clPlatformIDs[i];
		platform.id = platform_id;

		queryPlatformInfo(platform_id, CL_PLATFORM_NAME,	platform.name,		"CL_PLATFORM_NAME",		1024);
		queryPlatformInfo(platform_id, CL_PLATFORM_VENDOR,	platform.vendor,	"CL_PLATFORM_VENDOR",	1024);
		queryPlatformInfo(platform_id, CL_PLATFORM_VERSION,	platform.version,	"CL_PLATFORM_VERSION",	1024);

		platforms_.push_back(platform);
	}

	return true;
}

bool OpenCLEnum::queryDeviceInfo(Device &device)
{
	cl_device_id device_id = device.id;

	queryDeviceInfo(device_id, CL_DEVICE_TYPE,					device.device_type,		"CL_DEVICE_TYPE");
	queryDeviceInfo(device_id, CL_DEVICE_NAME,					device.name,			"CL_DEVICE_NAME", 1024);
	queryDeviceInfo(device_id, CL_DEVICE_VENDOR,				device.vendor,			"CL_DEVICE_VENDOR", 1024);
	queryDeviceInfo(device_id, CL_DEVICE_VENDOR_ID,				device.vendor_id,		"CL_DEVICE_VENDOR_ID");
	queryDeviceInfo(device_id, CL_DEVICE_VERSION,				device.version,			"CL_DEVICE_VERSION", 1024);
	queryDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,		device.compute_units,	"CL_DEVICE_MAX_COMPUTE_UNITS");
	queryDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE,		device.mem_size,		"CL_DEVICE_GLOBAL_MEM_SIZE");
	queryDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY,	device.clock,			"CL_DEVICE_MAX_CLOCK_FREQUENCY");
	queryDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,	device.max_workgroup_size, "CL_DEVICE_MAX_WORK_GROUP_SIZE");
	queryDeviceInfo(device_id, CL_DRIVER_VERSION,				device.driver_version,	"CL_DRIVER_VERSION", 1024);
	cl_bool is_unified_memory = CL_FALSE;
	queryDeviceInfo(device_id, CL_DEVICE_HOST_UNIFIED_MEMORY,	is_unified_memory,		"CL_DEVICE_HOST_UNIFIED_MEMORY");
	device.unified_memory = (is_unified_memory == CL_TRUE);

	std::set< std::string > extensions;
	queryExtensionList(device_id, extensions);

	if (extensions.count("cl_nv_device_attribute_query")) {
		queryDeviceInfo(device_id, CL_DEVICE_PCI_BUS_ID_NV,		device.nvidia_pci_bus_id,	"CL_DEVICE_PCI_BUS_ID_NV");
		queryDeviceInfo(device_id, CL_DEVICE_PCI_SLOT_ID_NV,	device.nvidia_pci_slot_id,	"CL_DEVICE_PCI_SLOT_ID_NV");
	}

	device.plain_name = device.name;

	if (device.device_type == CL_DEVICE_TYPE_GPU && device.vendor_id == gpu::VENDOR::ID_AMD && extensions.count(CL_AMD_DEVICE_ATTRIBUTE_QUERY_EXT)) {
		std::string amd_name;
		queryDeviceInfo(device_id, CL_DEVICE_BOARD_NAME_AMD,	amd_name,				"CL_DEVICE_BOARD_NAME_AMD", 1024);
		device.name = amd_name + " (" + device.name + ")";
		device.plain_name = amd_name;

		queryDeviceInfo(device_id, CL_DEVICE_TOPOLOGY_AMD,		device.device_topology_amd,	"CL_DEVICE_TOPOLOGY_AMD");
	}

	device.has_cl_khr_spir = (extensions.count("cl_khr_spir") != 0);

	return true;
}

template <typename T>
bool OpenCLEnum::queryDeviceInfo(cl_device_id device_id, unsigned int param, T &value, const std::string &param_name)
{
	cl_int res;
	OCL_TRACE(res = clGetDeviceInfo(device_id, param, sizeof(value), &value, NULL));
	if (res != CL_SUCCESS) {
		std::cerr << "clGetDeviceInfo(" << param_name << ") failed: " << ocl::errorString(res) << std::endl;
		return false;
	}
	return true;
}

bool OpenCLEnum::queryDeviceInfo(cl_device_id device_id, unsigned int param, std::string &value, const std::string &param_name, size_t max_size)
{
	cl_int res;
	if (max_size == 0) {
		OCL_TRACE(res = clGetDeviceInfo(device_id, param, 0, NULL, &max_size));
		if (res != CL_SUCCESS) {
			std::cerr << "clGetDeviceInfo(" << param_name << ") failed: " << ocl::errorString(res) << std::endl;
			return false;
		}
	}

	std::vector<char> data(max_size);
	OCL_TRACE(res = clGetDeviceInfo(device_id, param, max_size, data.data(), &max_size));
	if (res != CL_SUCCESS) {
		std::cerr << "clGetDeviceInfo(" << param_name << ") failed: " << ocl::errorString(res) << std::endl;
		return false;
	}

	value.assign(data.begin(), data.begin() + max_size);
	value = value.c_str();	// remove trailing null chars
	return true;
}

bool OpenCLEnum::queryPlatformInfo(cl_platform_id platform_id, unsigned int param, std::string &value, const std::string &param_name, size_t max_size)
{
	cl_int res;

	std::vector<char> data(max_size);
	OCL_TRACE(res = clGetPlatformInfo(platform_id, param, max_size, data.data(), &max_size));
	if (res != CL_SUCCESS) {
		std::cerr << "clGetPlatformInfo(" << param_name << ") failed: " << ocl::errorString(res) << std::endl;
		return false;
	}

	value.assign(data.begin(), data.begin() + max_size);
	value = value.c_str();	// remove trailing null chars
	return true;
}

bool OpenCLEnum::queryExtensionList(cl_device_id device_id, std::set<std::string> &extensions)
{
	std::string extensions_string;
	if (!queryDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, extensions_string, "CL_DEVICE_EXTENSIONS"))
		return false;

	std::vector<std::string> tokens = split(extensions_string, " ");
	extensions.insert(tokens.begin(), tokens.end());
	return true;
}

bool OpenCLEnum::enumDevices(cl_platform_id platform_id, bool silent)
{
	cl_int ciErrNum;

	cl_uint			uiNumDevices	= 0;		// number of devices available
	cl_device_type	device_type		= CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR;
	if (OCL_CPU_DEVICES_ENABLED) {
		device_type |= CL_DEVICE_TYPE_CPU;
	}

	OCL_TRACE(ciErrNum = clGetDeviceIDs(platform_id, device_type, 0, NULL, &uiNumDevices));
	if (ciErrNum != CL_SUCCESS && ciErrNum != CL_DEVICE_NOT_FOUND) {
		if (!silent) std::cerr << "clGetDeviceIDs failed: " << ocl::errorString(ciErrNum) << std::endl;
		return false;
	}

	if (ciErrNum == CL_DEVICE_NOT_FOUND || uiNumDevices == 0)
		return true;

	std::vector<cl_device_id> cdDevices(uiNumDevices);

	OCL_TRACE(ciErrNum = clGetDeviceIDs(platform_id, device_type, uiNumDevices, cdDevices.data(), NULL));
	if (ciErrNum != CL_SUCCESS) {
		if (!silent) std::cerr << "clGetDeviceIDs for " << uiNumDevices << " devices failed: " << ocl::errorString(ciErrNum) << std::endl;
		return false;
	}

	for (cl_uint i = 0; i < uiNumDevices; i++) {
		Device device;

		device.id			= cdDevices[i];
		device.platform_id	= platform_id;

		if (!queryDeviceInfo(device)) {
			if (!silent) std::cerr << device.name << ": can't query device info" << std::endl;
			continue;
		}

		if (device.max_workgroup_size < 256) {
			if (!silent) std::cout << device.name << ": too small max workgroup size " << device.max_workgroup_size << " (at least 256 required)" << std::endl;
			continue;
		}

#ifdef SPIR_SUPPORT
		if (!device.has_cl_khr_spir) {
	#ifdef CUDA_SUPPORT
			if (device.vendor_id != gpu::VENDOR::ID_NVIDIA && device.vendor.find("NVIDIA") == std::string::npos) {
	#endif
				if (!silent) std::cout << device.name << ": no SPIR support" << std::endl;
	#ifdef CUDA_SUPPORT
			}
	#endif
		}
#endif

		{
			// Blacklist of bad drivers
			if (device.name == "Microsoft Basic Render Driver")
				continue;
		}

		devices_.push_back(device);
	}

	return true;
}

bool OpenCLEnum::compareDevice(const Device &dev1, const Device &dev2)
{
	if (dev1.name	> dev2.name)	return false;
	if (dev1.name	< dev2.name)	return true;
	if (dev1.id		> dev2.id)		return false;
	return true;
}

bool OpenCLEnum::enumDevices(bool silent)
{
	if (!ocl_init()) {
		if (!silent) std::cerr << "Can't load OpenCL library" << std::endl;
		return false;
	}

	if (!enumPlatforms(silent))
		return false;

	for (size_t k = 0; k < platforms_.size(); k++) {
		if (!enumDevices(platforms_[k].id, silent)) {
			if (!silent) std::cerr << platforms_[k].name << ": can't enumerate devices" << std::endl;
		}
	}

	std::sort(devices_.begin(), devices_.end(), compareDevice);

	return true;
}

std::shared_ptr<ocl::OpenCLEngine> OpenCLEnum::Device::createEngine(bool print_info)
{
	std::shared_ptr<ocl::OpenCLEngine> engine(new ocl::OpenCLEngine());
	engine->init(platform_id, id, 0, print_info);
	return engine;
}
