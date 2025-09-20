#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <iomanip>

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

	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

constexpr std::size_t operator"" _KB(unsigned long long v) { return v * 1024ull; }
constexpr std::size_t operator"" _MB(unsigned long long v) { return v * 1024ull * 1024ull; }
constexpr std::size_t operator"" _GB(unsigned long long v) { return v * 1024ull * 1024ull * 1024ull; }


inline void printLabel(const std::string& label, size_t offset = 0) {
    std::string indent(offset, ' ');
    std::cout << "\n" << indent << "=== " << label << " ===\n";
}

template<typename T>
inline void printField(const std::string& label, T value,
                       size_t offset = 0, int labelWidth = 30,
                       const std::string& suffix = "") {
    std::string indent(offset, ' ');
    std::cout << std::left
              << indent << std::setw(labelWidth) << (label + ":")
              << value << (suffix.empty() ? "" : " " + suffix) << "\n";
}

std::string deviceTypeToString(cl_device_type type) {
    switch (type) {
        case CL_DEVICE_TYPE_CPU: return "CPU";
        case CL_DEVICE_TYPE_GPU: return "GPU";
        case CL_DEVICE_TYPE_ACCELERATOR: return "ACCELERATOR";
        case CL_DEVICE_TYPE_DEFAULT: return "DEFAULT";
        case CL_DEVICE_TYPE_CUSTOM: return "CUSTOM";
        default: return "UNKNOWN";
    }
}

template<typename T>
T getClInfoValue(cl_device_id device, cl_device_info param) {
    T value;
    OCL_SAFE_CALL(clGetDeviceInfo(device, param, sizeof(T), &value, nullptr));
    return value;
}

template <typename F, typename Id>
std::string getClInfoStringImpl(Id id, cl_uint param, F clGetInfoFn) {
    size_t size = 0;
    OCL_SAFE_CALL(clGetInfoFn(id, param, 0, nullptr, &size));
    std::vector<char> buffer(size);
    OCL_SAFE_CALL(clGetInfoFn(id, param, size, buffer.data(), nullptr));
    return buffer.data();
}

std::string getClInfoString(cl_platform_id platform, cl_platform_info param) {
    return getClInfoStringImpl(platform, param, clGetPlatformInfo);
}

std::string getClInfoString(cl_device_id device, cl_device_info param) {
    return getClInfoStringImpl(device, param, clGetDeviceInfo);
}

struct DeviceInfo {
    explicit DeviceInfo(cl_device_id device) {
        device_name = getClInfoString(device, CL_DEVICE_NAME);
        device_type = getClInfoValue<cl_device_type>(device, CL_DEVICE_TYPE);
        device_global_mem_size = getClInfoValue<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
        device_max_mem_alloc_size = getClInfoValue<cl_ulong>(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
        device_max_work_group_size = getClInfoValue<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        driver_version = getClInfoString(device, CL_DRIVER_VERSION);
        device_address_bits = getClInfoValue<cl_uint>(device, CL_DEVICE_ADDRESS_BITS);
        device_max_constant_buffer_size = getClInfoValue<cl_ulong>(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
    }

    std::string device_name;
    cl_device_type device_type;
    cl_ulong device_global_mem_size;
    cl_ulong device_max_mem_alloc_size;
    size_t device_max_work_group_size;
    std::string driver_version;
    cl_uint device_address_bits;
    cl_ulong device_max_constant_buffer_size;
};

void printDeviceInfo(const DeviceInfo& info, size_t offset_size = 0, int labelWidth = 30) {
    printField("Device name", info.device_name, offset_size, labelWidth);
    printField("Device type", deviceTypeToString(info.device_type), offset_size, labelWidth);
    printField("Global mem size", info.device_global_mem_size / 1_MB, offset_size, labelWidth, "MB");
    printField("Max mem alloc size", info.device_max_mem_alloc_size / 1_MB, offset_size, labelWidth, "MB");
    printField("Max work-group size", info.device_max_work_group_size, offset_size, labelWidth);
    printField("Driver version", info.driver_version, offset_size, labelWidth);
    printField("Address bits", info.device_address_bits, offset_size, labelWidth);
    printField("Max constant buffer size", info.device_max_constant_buffer_size / 1_KB, offset_size, labelWidth, "KB");
}

void printPlatformInfo(cl_platform_id platform, size_t offset_size = 0, int labelWidth = 30) {
    printField("Platform name", getClInfoString(platform, CL_PLATFORM_NAME), offset_size, labelWidth);
    printField("Platform vendor", getClInfoString(platform, CL_PLATFORM_VENDOR), offset_size, labelWidth);
}

std::vector<cl_device_id> getDeviceIds(cl_platform_id platform) {
    cl_uint devicesCount = 0;
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
    std::vector<cl_device_id> devices(devicesCount);
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), &devicesCount));
    return devices;
}

std::vector<cl_platform_id> getPlatformsIds() {
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));
    return platforms;
}


int main() {
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    const size_t globalLabelWidth = 50;
    const size_t offsetStep = 4;

    auto platforms = getPlatformsIds();
    printField("Number of OpenCL platforms", platforms.size(), 0, globalLabelWidth);

    for (size_t p = 0; p < platforms.size(); ++p) {
        printLabel("Platform " + std::to_string(p + 1) + "/" + std::to_string(platforms.size()));
        printPlatformInfo(platforms[p], 1 * offsetStep, globalLabelWidth - offsetStep);

        auto devices = getDeviceIds(platforms[p]);
        printField("Number of devices", devices.size(), 1 * offsetStep, globalLabelWidth - offsetStep);

        for (size_t d = 0; d < devices.size(); ++d) {
            printLabel("Device " + std::to_string(d + 1) + "/" + std::to_string(devices.size()), 2 * offsetStep);
            printDeviceInfo(DeviceInfo(devices[d]), 3 * offsetStep, globalLabelWidth - 3 * offsetStep);
        }
    }
}
