#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/fast_random.h>
#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_matrix01TransposeNaive(ocl::getMatrix01TransposeNaive());
    ocl::KernelSource ocl_matrix02TransposeCoalescedViaLocalMemory(ocl::getMatrix02TransposeCoalescedViaLocalMemory());

    unsigned int ksize = 512;
    unsigned int w = ksize * 32;
    unsigned int h = ksize * 16;
    std::cout << "Matrix size: rows=H=" << h << " x cols=W=" << w << " (" << sizeof(float) * w * h / 1024 / 1024 << " MB)" << std::endl;

    std::vector<float> input_cpu(h * w, 0);
    FastRandom r;
    for (size_t i = 0; i < h * w; ++i) {
        input_cpu[i] = r.nextf();
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32f input_matrix_gpu(h * w); // rows=H x cols=W
    gpu::gpu_mem_32f output_matrix_gpu(w * h); // rows=W x cols=H

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_matrix_gpu.writeN(input_cpu.data(), h * w);

    std::vector<std::string> algorithm_names = {
        "01 naive transpose (non-coalesced)",
        "02 transpose via local memory (coalesced)",
    };

    for (size_t algorithm_index = 0; algorithm_index < algorithm_names.size(); ++algorithm_index) {
        const std::string& algorithm = algorithm_names[algorithm_index];
        std::cout << "______________________________________________________" << std::endl;
        std::cout << "Evaluating algorithm #" << (algorithm_index + 1) << "/" << algorithm_names.size() << ": " << algorithm << std::endl;

        // Запускаем алгоритм (несколько раз и с замером времени выполнения)
        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;

            if (algorithm == "01 naive transpose (non-coalesced)") {
                ocl_matrix01TransposeNaive.exec(gpu::WorkSize(1, 1, w, h), input_matrix_gpu, output_matrix_gpu, w, h);
            } else if (algorithm == "02 transpose via local memory (coalesced)") {
                ocl_matrix02TransposeCoalescedViaLocalMemory.exec(gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, w, h), input_matrix_gpu, output_matrix_gpu, w, h);
            } else {
                rassert(false, 652345234321, algorithm, algorithm_index);
            }

            times.push_back(t.elapsed());
        }
        std::cout << "algorithm times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        // Вычисляем достигнутую эффективную пропускную способность алгоритма (из соображений что мы отработали в один проход по входному массиву)
        double memory_size_gb = 2.0 * sizeof(float) * w * h / 1024.0 / 1024.0 / 1024.0;
        std::cout << "median effective algorithm bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

        // Сверяем результат
        std::vector<float> results = output_matrix_gpu.readVector(); // input matrix: w x h -> output matrix: h x w
        for (size_t j = 0; j < h; ++j) {
            for (size_t i = 0; i < w; ++i) {
                rassert(results[i * h + j] == input_cpu[j * w + i], 6573452432, i, j);
            }
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