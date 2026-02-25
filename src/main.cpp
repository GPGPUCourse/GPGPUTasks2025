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

    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
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
        
        // TODO 1.1
        // Проверил: если передать 239 вместо CL_PLATFORM_NAME, программа ожидаемо падает.
        // Выдает ошибку: "OpenCL error code -30". 
        // Открыл cl.h на строке 103, там сказано: #define CL_INVALID_VALUE -30.
        // В документации Khronos написано, что ошибка CL_INVALID_VALUE возвращается, если параметр param_name
        // не является одним из поддерживаемых значений. Поэтому возвращаю правильный CL_PLATFORM_NAME обратно.

        // TODO 1.2
        // Вектор делаю типа char, а не unsigned char, чтобы cout нормально выводил текст, а не ругался
        std::vector<char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3: Вендор платформы
        size_t vendorSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorSize));
        std::vector<char> platformVendor(vendorSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorSize, platformVendor.data(), nullptr));
        std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

        // TODO 2.1: Запрашиваем число доступных устройств
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

        // Выделяем память и получаем сами устройства
        std::vector<cl_device_id> devices(devicesCount);
        if (devicesCount > 0) {
            OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
        }

        for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
        {
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id device = devices[deviceIndex];

            // TODO 2.2: Свойства устройства
            
            // 1. Название устройства
            size_t deviceNameSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
            std::vector<char> deviceName(deviceNameSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
            std::cout << "        Device name: " << deviceName.data() << std::endl;

            // 2. Тип устройства (используем побитовое 'И', т.к. тип - это битовая маска)
            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
            std::string typeString;
            if (deviceType & CL_DEVICE_TYPE_GPU) typeString += "GPU ";
            if (deviceType & CL_DEVICE_TYPE_CPU) typeString += "CPU ";
            if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) typeString += "Accelerator ";
            if (typeString.empty()) typeString = "Unknown";
            std::cout << "        Device type: " << typeString << std::endl;

            // 3. Размер памяти в мегабайтах (cl_ulong возвращает байты, делим на 1024^2)
            cl_ulong memSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memSize, nullptr));
            std::cout << "        Memory size: " << (memSize / (1024 * 1024)) << " MB" << std::endl;

            // 4. Дополнительное свойство: Версия OpenCL
            size_t versionSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, nullptr, &versionSize));
            std::vector<char> deviceVersion(versionSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, versionSize, deviceVersion.data(), nullptr));
            std::cout << "        OpenCL version: " << deviceVersion.data() << std::endl;

            // 5. Дополнительное свойство: Максимальное количество вычислительных блоков (Compute Units)
            cl_uint computeUnits = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &computeUnits, nullptr));
            std::cout << "        Max compute units: " << computeUnits << std::endl;
        }
    }

    return 0;
}