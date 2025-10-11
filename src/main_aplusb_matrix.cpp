#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>

void runKernel(
    ocl::KernelSource kernel,
    unsigned int width,
    unsigned int height,
    const std::vector<unsigned int>& as,
    const std::vector<unsigned int>& bs,
    std::vector<unsigned int>& cs,
    const gpu::gpu_mem_32u& a_gpu,
    const gpu::gpu_mem_32u& b_gpu,
    const gpu::gpu_mem_32u& c_gpu)
{
    size_t length = width * height;

    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        gpu::WorkSize workSize(GROUP_SIZE, 1, width, height);
        kernel.exec(workSize, a_gpu, b_gpu, c_gpu, width, height);
        times.push_back(t.elapsed());
    }
    std::cout << "a + b matrix kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    double memory_size_gb = sizeof(unsigned int) * 3 * length / 1024.0 / 1024.0 / 1024.0;
    std::cout << "a + b kernel median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

    c_gpu.readN(cs.data(), length);
    for (size_t i = 0; i < length; ++i) {
        rassert(cs[i] == as[i] + bs[i], 321418230421312512, cs[i], as[i] + bs[i], i);
    }
}
void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_aplusb_matrix_bad(ocl::getAplusBMatrixBad());
    ocl::KernelSource ocl_aplusb_matrix_good(ocl::getAplusBMatrixGood());

    avk2::KernelSource vk_aplusb_matrix_bad(avk2::getAplusBMatrixBad());
    avk2::KernelSource vk_aplusb_matrix_good(avk2::getAplusBMatrixGood());

    unsigned int task_size = 64;
    unsigned int width = task_size * 256;
    unsigned int height = task_size * 128;
    std::cout << "matrices size: " << width << "x" << height << " = 3 * " << (sizeof(unsigned int) * width * height / 1024 / 1024) << " MB" << std::endl;

    std::vector<unsigned int> as(width * height, 0);
    std::vector<unsigned int> bs(width * height, 0);
    std::vector<unsigned int> cs(width * height, 0);
    for (size_t i = 0; i < width * height; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }

    gpu::gpu_mem_32u a_gpu(width * height), b_gpu(width * height), c_gpu(width * height);
    a_gpu.writeN(as.data(), width * height);
    b_gpu.writeN(bs.data(), width * height);

    {
        std::cout << "Running BAD matrix kernel..." << std::endl;
        runKernel(ocl_aplusb_matrix_bad, width, height, as, bs, cs, a_gpu, b_gpu, c_gpu);
    }

    {
        std::cout << "Running GOOD matrix kernel..." << std::endl;
        runKernel(ocl_aplusb_matrix_good, width, height, as, bs, cs, a_gpu, b_gpu, c_gpu);
    }
}

int main(int argc, char** argv)
{
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за выбора CUDA API (его нет на процессоре - т.е. в случае CI на GitHub Actions)
            return 0;
        }
        if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за того что задание еще не выполнено
            return 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            return 1;
        }
    }

    return 0;
}
