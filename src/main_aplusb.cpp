#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_aplusb(ocl::getAplusB());

    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> bs(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }

    gpu::gpu_mem_32u a_gpu(n), b_gpu(n), c_gpu(n);

    a_gpu.writeN(as.data(), n);
    b_gpu.writeN(bs.data(), n);

    gpu::WorkSize workSize(GROUP_SIZE, n);
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        ocl_aplusb.exec(workSize, a_gpu, b_gpu, c_gpu, n);

        times.push_back(t.elapsed());
    }
    std::cout << "a + b kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти
    double memory_size_gb = sizeof(unsigned int) * 3 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "a + b kernel median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

    std::vector<unsigned int> cs(n, 0);
    c_gpu.readN(cs.data(), n);

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(cs[i] == as[i] + bs[i], 321418230421312, cs[i], as[i] + bs[i], i);
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
        } if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за того что задание еще не выполнено
            return 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            return 1;
        }
    }

    return 0;
}
