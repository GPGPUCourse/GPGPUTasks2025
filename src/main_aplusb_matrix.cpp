#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>

static inline unsigned int round_up(unsigned int v, unsigned int multiple) {
    return ((v + multiple - 1) / multiple) * multiple;
}

void run(int argc, char** argv)
{
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    // TODO 100 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 100 после этого реализуйте два кернела - максимально эффективный и максимально неэффктивный вариант сложения матриц - src/kernels/<ваш выбор>/aplusb_matrix_<bad/good>.<ваш выбор>
    // TODO 100 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DCUDA_SUPPORT=ON в CMake options
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

    unsigned int task_size = 64;
    unsigned int width = task_size * 256;
    unsigned int height = task_size * 128;
    std::cout << "matrices size: " << width << "x" << height << " = 3 * " << (sizeof(unsigned int) * width * height / 1024 / 1024) << " MB" << std::endl;

    std::vector<unsigned int> as(width * height, 0);
    std::vector<unsigned int> bs(width * height, 0);
    for (size_t i = 0; i < width * height; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u a_gpu(width * height), b_gpu(width * height), c_gpu(width * height);

    a_gpu.writeN(as.data(), width * height);
    b_gpu.writeN(bs.data(), width * height);

/* проводил эксперименты с разными workSize. Исходя из того, как я понял теорию, хуже всего должно быть при
 * 256x1 потому что при 32x8 и тем более 16x16 что-то попадает в кеш. Я почитал, в реальности же похоже, что
 * разница в производительности будет настолько мала, что из-за погрешности таймера ее просто не будет видно.
 * Это так? Если обратите внимание на этот комментарий, ответьте, пожалуйста, после ревью.
 * */
//    for (int i = 0; i < 3; i++) {
    for (int i = 0; i < 1; i++) {
        {
            std::cout << "Running BAD matrix kernel..." << std::endl;

            // Запускаем кернел (несколько раз и с замером времени выполнения)
            std::vector<double> times;
            for (int iter = 0; iter < 10; ++iter) {
                timer t;

                // Настраиваем размер рабочего пространства (n) и размер рабочих групп в этом рабочем пространстве (GROUP_SIZE=256)
                // Обратите внимание что сейчас указана рабочая группа размера 1х1 в рабочем пространстве width x height, это не то что вы хотите
                // TODO И в плохом и в хорошем кернеле рабочая группа обязана состоять из 256 work-items
                unsigned int localX = 256, localY = 1;
                unsigned int globalX = round_up(width, 256), globalY = round_up(height, 1);
                switch(i)
                {
                    case 0:
                        localX = 256; localY = 1;
                        globalX = round_up(width, 256);
                        globalY = round_up(height, 1);
                        break;
                    case 1:
                        localX = 32; localY = 8;
                        globalX = round_up(width, 32);
                        globalY = round_up(height, 8);
                        break;
                    case 2:
                        localX = 16; localY = 16;
                        globalX = round_up(width, 16);
                        globalY = round_up(height, 16);
                        break;
                    default:
                        localX = 256; localY = 1;
                        globalX = round_up(width, 256);
                        globalY = round_up(height, 1);
                        break;
                }
                gpu::WorkSize workSize(localX, localY, globalX, globalY);
                ocl_aplusb_matrix_bad.exec(workSize, a_gpu, b_gpu, c_gpu, width, height);

                times.push_back(t.elapsed());
            }
            std::cout << "a + b matrix (BAD) kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

            double memory_size_gb = double(sizeof(unsigned int)) * 3.0 * double(width) * double(height) / 1024.0 / 1024.0 / 1024.0;
            std::cout << "a + b matrix (BAD) median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

            std::vector<unsigned int> cs(width * height, 0);
            c_gpu.readN(cs.data(), width * height);

            // Сверяем результат
            for (size_t i = 0; i < width * height; ++i) {
                rassert(cs[i] == as[i] + bs[i], 321418230421312512, cs[i], as[i] + bs[i], i);
            }
        }

        {
            std::cout << "Running GOOD matrix kernel..." << std::endl;
            // Запускаем кернел (несколько раз и с замером времени выполнения)
            std::vector<double> times;
            for (int iter = 0; iter < 10; ++iter) {
                timer t;

                // Настраиваем размер рабочего пространства (n) и размер рабочих групп в этом рабочем пространстве (GROUP_SIZE=256)
                // Обратите внимание что сейчас указана рабочая группа размера 1х1 в рабочем пространстве width x height, это не то что вы хотите
                // TODO И в плохом и в хорошем кернеле рабочая группа обязана состоять из 256 work-items
                unsigned int localX = 256, localY = 1;
                unsigned int globalX = round_up(width, 256), globalY = round_up(height, 1);
                switch(i)
                {
                    case 0:
                        localX = 256; localY = 1;
                        globalX = round_up(width, 256);
                        globalY = round_up(height, 1);
                        break;
                    case 1:
                        localX = 32; localY = 8;
                        globalX = round_up(width, 32);
                        globalY = round_up(height, 8);
                        break;
                    case 2:
                        localX = 16; localY = 16;
                        globalX = round_up(width, 16);
                        globalY = round_up(height, 16);
                        break;
                    default:
                        localX = 256; localY = 1;
                        globalX = round_up(width, 256);
                        globalY = round_up(height, 1);
                        break;
                }
                gpu::WorkSize workSize(localX, localY, globalX, globalY);
                ocl_aplusb_matrix_good.exec(workSize, a_gpu, b_gpu, c_gpu, width, height);

                times.push_back(t.elapsed());
            }
            std::cout << "a + b matrix (GOOD) kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

            double memory_size_gb = double(sizeof(unsigned int)) * 3.0 * double(width) * double(height) / 1024.0 / 1024.0 / 1024.0;
            std::cout << "a + b matrix (GOOD) median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

            std::vector<unsigned int> cs(width * height, 0);
            c_gpu.readN(cs.data(), width * height);

            // Сверяем результат
            for (size_t i = 0; i < width * height; ++i) {
                rassert(cs[i] == as[i] + bs[i], 321418230365731436, cs[i], as[i] + bs[i], i);
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
