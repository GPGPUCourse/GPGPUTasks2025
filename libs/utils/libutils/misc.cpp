#include "misc.h"

#include <libgpu/vulkan/device.h>
#include <libgpu/vulkan/vulkan_api_headers.h>

#ifdef CUDA_SUPPORT
#include <cuda_runtime_api.h>
#endif

void gpu::printDeviceInfo(const gpu::Device& device)
{
    {
        std::cout << "API: ";
        int count = 0;
#ifdef CUDA_SUPPORT
        if (device.supports_cuda) {
            std::cout << ((count > 0) ? "+" : "") << "CUDA";
            ++count;
        }
#endif
        if (device.supports_opencl) {
            std::cout << ((count > 0) ? "+" : "") << "OpenCL";
            ++count;
        }
        if (device.supports_vulkan) {
            std::cout << ((count > 0) ? "+" : "") << "Vulkan";
            ++count;
        }
        std::cout << ". ";
    }

#ifdef CUDA_SUPPORT
    if (device.supports_cuda) {
        int driverVersion = 239;
        cudaDriverGetVersion(&driverVersion);
        std::cout << "GPU. " << device.name << " (CUDA " << driverVersion << ").";
    } else
#endif
        if (device.supports_opencl) {
        ocl::DeviceInfo info;
        info.init(device.device_id_opencl);
        if (info.device_type == CL_DEVICE_TYPE_GPU) {
            std::cout << "GPU.";
        } else if (info.device_type == CL_DEVICE_TYPE_CPU) {
            std::cout << "CPU.";
        } else {
            throw std::runtime_error(
                "Only CPU and GPU supported! But type=" + to_string(info.device_type) + " encountered!");
        }
        std::cout << " " << info.device_name << ".";
        if (info.device_type == CL_DEVICE_TYPE_CPU) {
            std::cout << " " << info.vendor_name << ".";
        }
    } else if (device.supports_vulkan) {
        avk2::Device info(device.device_id_vulkan);
        info.init(true);
        if ((vk::PhysicalDeviceType)info.device_type == vk::PhysicalDeviceType::eDiscreteGpu) {
            std::cout << "GPU.";
        } else if ((vk::PhysicalDeviceType)info.device_type == vk::PhysicalDeviceType::eIntegratedGpu) {
            std::cout << "iGPU.";
        } else if ((vk::PhysicalDeviceType)info.device_type == vk::PhysicalDeviceType::eVirtualGpu) {
            std::cout << "vGPU.";
        } else if ((vk::PhysicalDeviceType)info.device_type == vk::PhysicalDeviceType::eCpu) {
            std::cout << "CPU.";
        } else {
            throw std::runtime_error(
                "Only CPU and GPU supported! But type=" + to_string(info.device_type) + " encountered!");
        }
        std::cout << " " << info.name << ".";
        if (info.device_type == CL_DEVICE_TYPE_CPU) {
            std::cout << " " << info.vendor_name << ".";
        }
    } else {
        rassert(false, 4356234512341, device.name);
    }

    if (device.supportsFreeMemoryQuery()) {
        std::cout << " Free memory: " << (device.getFreeMemory() >> 20) << "/" << (device.mem_size >> 20) << " Mb.";
    } else {
        std::cout << " Total memory: " << (device.mem_size >> 20) << " Mb.";
    }

    std::cout << std::endl;
}

gpu::Device gpu::chooseGPUDevice(const std::vector<gpu::Device>& devices, int argc, char** argv)
{
    unsigned int device_index = std::numeric_limits<unsigned int>::max();

    if (devices.size() == 0) {
        throw std::runtime_error("No GPU devices found!");
    } else {
        std::cout << "Available devices:" << std::endl;
        for (int i = 0; i < devices.size(); ++i) {
            std::cout << "  Device #" << i << ": ";
            gpu::printDeviceInfo(devices[i]);
        }
        if (devices.size() == 1) {
            device_index = 0;
        } else {
            if (argc != 2) {
                std::cerr << "Usage: <app> <DeviceIndex>" << std::endl;
                std::cerr << "	Where <DeviceIndex> should be from 0 to " << (devices.size() - 1) << " (inclusive)" << std::endl;
                throw std::runtime_error("Illegal arguments!");
            } else {
                device_index = atoi(argv[1]);
                rassert(to_string(device_index) == argv[1], 4365245324312, "Unexpected non-integer argument:", argv[1]);
                if (device_index >= devices.size()) {
                    std::cerr << "<DeviceIndex> should be from 0 to " << (devices.size() - 1) << " (inclusive)! But " << argv[1] << " provided!" << std::endl;
                    throw std::runtime_error("Illegal arguments!");
                }
            }
        }
        std::cout << "Using device #" << device_index << ": ";
        gpu::printDeviceInfo(devices[device_index]);
    }
    return devices[device_index];
}

gpu::Context gpu::activateContext(const gpu::Device& device, gpu::Context::Type api)
{
    gpu::Context context;

    if (api == gpu::Context::TypeOpenCL) {
        if (!device.supports_opencl) {
            std::cout << "Device " << device.name << " doesn't support OpenCL" << std::endl;
            throw std::runtime_error(DEVICE_NOT_SUPPORT_API);
        }
        std::cout << "Using OpenCL API..." << std::endl;
        context.init(device.device_id_opencl);
    } else if (api == gpu::Context::TypeCUDA) {
#ifndef CUDA_SUPPORT
        // TODO 000 если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DCUDA_SUPPORT=ON в CMake options
        rassert(false, "To use CUDA you need to enable CUDA via CMake options: -DCUDA_SUPPORT=ON");
#endif
        if (!device.supports_cuda) {
            std::cout << "Device " << device.name << " doesn't support CUDA" << std::endl;
            throw std::runtime_error(DEVICE_NOT_SUPPORT_API);
        }
        std::cout << "Using CUDA API..." << std::endl;
        context.init(device.device_id_cuda);
    } else if (api == gpu::Context::TypeVulkan) {
        if (!device.supports_vulkan) {
            std::cout << "Device " << device.name << " doesn't support Vulkan" << std::endl;
            throw std::runtime_error(DEVICE_NOT_SUPPORT_API);
        }
        std::cout << "Using Vulkan API..." << std::endl;
        context.initVulkan(device.device_id_vulkan);
        // context.setVKValidationLayers(true); // раскоментируйте если хотите воспользоваться debugPrintfEXT
    } else {
        rassert(false, api);
    }

    context.activate();
    return context;
}
