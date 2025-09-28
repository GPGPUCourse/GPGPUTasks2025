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

    gpu::gpu_mem_32u a_gpu(width * height), b_gpu(width * height), c_gpu(width * height);
    a_gpu.writeN(as.data(), as.size());
    b_gpu.writeN(bs.data(), bs.size());

    {
        std::cout << "Running BAD matrix kernel..." << std::endl;

        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;

            // Настраиваем размер рабочего пространства (n) и размер рабочих групп в этом рабочем пространстве (GROUP_SIZE=256)
            // Обратите внимание что сейчас указана рабочая группа размера 1х1 в рабочем пространстве width x height, это не то что вы хотите
            // TODO И в плохом и в хорошем кернеле рабочая группа обязана состоять из 256 work-items
            gpu::WorkSize workSize(GROUP_SIZE_X, GROUP_SIZE_Y, width, height);

            // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
            // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
            // TODO раскомментируйте вызов вашего API и поправьте его
            if (context.type() == gpu::Context::TypeOpenCL) {
                ocl_aplusb_matrix_bad.exec(workSize, a_gpu, b_gpu, c_gpu, width, height);
            } else if (context.type() == gpu::Context::TypeCUDA) {
                // cuda::aplusb_matrix_bad(workSize, a_gpu, ...);
            } else if (context.type() == gpu::Context::TypeVulkan) {
                struct {
                    unsigned int width;
                    unsigned int height;
                } params = { width, height };
                // vk_aplusb_matrix_bad.exec(params, workSize, a_gpu, ...);
            } else {
                rassert(false, 4531412341, context.type());
            }

            times.push_back(t.elapsed());
        }
        std::cout << "a + b matrix kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        // TODO Удалите этот rassert - вычислите достигнутую эффективную пропускную способность видеопамяти
        // rassert(false, 54623414231);

        // TODO Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
        std::vector<unsigned int> cs(width * height, 0);
        c_gpu.readN(cs.data(), cs.size());

        // Сверяем результат
        for (size_t i = 0; i < width * height; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230421312512, cs[i], as[i] + bs[i], i);
        }
    }

    {
        std::cout << "Running GOOD matrix kernel..." << std::endl;
        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;
            gpu::WorkSize workSize(GROUP_SIZE_X, GROUP_SIZE_Y, width, height);

            if (context.type() == gpu::Context::TypeOpenCL) {
                ocl_aplusb_matrix_good.exec(workSize, a_gpu, b_gpu, c_gpu, width, height);
            } else {
                rassert(false, 4531412341, context.type());
            }

            times.push_back(t.elapsed());
        }
        std::cout << "a + b matrix kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        // TODO Почти тот же код что с плохим кернелом, но теперь с хорошим, рекомендуется копи-паста

        // TODO Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
        std::vector<unsigned int> cs(width * height, 0);
        c_gpu.readN(cs.data(), cs.size());

        // Сверяем результат
        for (size_t i = 0; i < width * height; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230365731436, cs[i], as[i] + bs[i], i);
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
