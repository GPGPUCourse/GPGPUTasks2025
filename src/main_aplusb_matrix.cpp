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
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_aplusb_matrix_bad(ocl::getAplusBMatrixBad());
    ocl::KernelSource ocl_aplusb_matrix_good(ocl::getAplusBMatrixGood());

    avk2::KernelSource vk_aplusb_matrix_bad(avk2::getAplusBMatrixBad());
    avk2::KernelSource vk_aplusb_matrix_good(avk2::getAplusBMatrixGood());

    constexpr unsigned task_size = 64;
    constexpr unsigned width = task_size * 256;
    constexpr unsigned height = task_size * 128;
    std::cout << "matrices size: " << width << "x" << height << " = 3 * " << (sizeof(unsigned int) * width * height / 1024 / 1024) << " MB" << std::endl;

    std::vector<unsigned int> as(width * height, 0);
    std::vector<unsigned int> bs(width * height, 0);
    for (size_t i = 0; i < width * height; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }

    // Аллоцируем буферы в VRAM
    constexpr unsigned len = width * height;
    gpu::gpu_mem_32u a_gpu(len), b_gpu(len), c_gpu(len);

    a_gpu.writeN(as.data(), as.size());
    b_gpu.writeN(bs.data(), bs.size());

    auto div_up = [](unsigned x, unsigned y) {
        return (x + y - 1) / y;
    };

    constexpr unsigned work_group_dim_x = 16, work_group_dim_y = 16;
    constexpr double GB = 1024 * 1024 * 1024;

    {
        std::cout << "Running BAD matrix kernel..." << std::endl;

        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;
            gpu::WorkSize work_size(work_group_dim_x, work_group_dim_y, div_up(width, work_group_dim_x) * work_group_dim_x, div_up(height, work_group_dim_y) * work_group_dim_y);
            ocl_aplusb_matrix_bad.exec(work_size, a_gpu, b_gpu, c_gpu, width, height);
            times.push_back(t.elapsed());
        }

        std::cout << "a_gpu + b_gpu matrix kernel times - " << stats::valuesStatsLine(times) << std::endl;

        double used_memory_gb = sizeof(unsigned) * 3 * len / GB;

        double median_time = stats::median(times);
        double bandwidth = used_memory_gb / median_time;

        std::cout
            << "a_gpu + b_gpu matrix kernel median bandwidth: "
            << bandwidth << " GB/s"
            << std::endl;
        std::vector<unsigned> cs(len, 0);
        c_gpu.readN(cs.data(), cs.size());
        for (size_t i = 0; i < len; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230421312512, cs[i], as[i] + bs[i], i);
        }
    }

    {
        std::cout << "Running GOOD matrix kernel..." << std::endl;

        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;
            gpu::WorkSize work_size(work_group_dim_x, work_group_dim_y, div_up(width, work_group_dim_x) * work_group_dim_x, div_up(height, work_group_dim_y) * work_group_dim_y);
            ocl_aplusb_matrix_good.exec(work_size, a_gpu, b_gpu, c_gpu, width, height);
            times.push_back(t.elapsed());
        }

        std::cout << "a_gpu + b_gpu matrix kernel times - " << stats::valuesStatsLine(times) << std::endl;

        double used_memory_gb = sizeof(unsigned) * 3 * len / GB;

        double median_time = stats::median(times);
        double bandwidth = used_memory_gb / median_time;

        std::cout
            << "a_gpu + b_gpu matrix kernel median bandwidth: "
            << bandwidth << " GB/s"
            << std::endl;
        std::vector<unsigned> cs(len, 0);
        c_gpu.readN(cs.data(), cs.size());
        for (size_t i = 0; i < len; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230421312512, cs[i], as[i] + bs[i], i);
        }
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
