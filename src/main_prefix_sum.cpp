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

    // TODO 000 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 000 после этого изучите этот код, запустите его, изучите соответсвующий вашему выбору кернел - src/kernels/<ваш выбор>/aplusb.<ваш выбор>
    // TODO 000 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DCUDA_SUPPORT=ON в CMake options
    // TODO 010 P.S. так же в случае CUDA - добавьте в CMake options (НЕ меняйте сами CMakeLists.txt чтобы не менять окружение тестирования):
    // TODO 010 "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo" (первое - чтобы включить поддержку WMMA, второе - чтобы compute-sanitizer и профилировщик знали номера строк кернела)
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_fill_with_zeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_sum_reduction(ocl::getPrefixSum01Reduction());
    ocl::KernelSource ocl_prefix_accumulation(ocl::getPrefixSum02PrefixAccumulation());

    avk2::KernelSource vk_fill_with_zeros(avk2::getFillBufferWithZeros());
    avk2::KernelSource vk_sum_reduction(avk2::getPrefixSum01Reduction());
    avk2::KernelSource vk_prefix_accumulation(avk2::getPrefixSum02PrefixAccumulation());

    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    size_t total_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        as[i] = (3 * (i + 5) + 7) % 17;
        total_sum += as[i];
        rassert(total_sum < std::numeric_limits<unsigned int>::max(), 5462345234231, total_sum, as[i], i); // ensure no overflow
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n), buffer1_pow2_sum_gpu(n), buffer2_pow2_sum_gpu(n), prefix_sum_accum_gpu(n);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
        // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
        if (context.type() == gpu::Context::TypeOpenCL) {
            // TODO
            // ocl_fill_with_zeros.exec();
            // ocl_sum_reduction.exec();
            // ocl_prefix_accumulation.exec();

            // идея решения - в 01_reduction я параллельно уменьшаю размер массива вдвое сохраняя суммы пар в новый массив
            // в 02_prefix_accumulation я нахожу какие элементы относятся ко второй половине каждого блока и добавляю им сумму блока слева
            // так шаг за шагом я разгоняю эти блоковые суммы по всему массиву пока каждый элемент не получит свой префикс
            // то есть тут я в цикле по stage каждый раз обновляю prefix_sum_accum через ocl_prefix_accumulation
            // затем через ocl_sum_reduction строю новый более короткий массив блоковых сумм
            // после этого меняю буферы местами обновляю active и перехожу к следующему шагу с большим размером блока пока active не станет равен 1 и не останется одна блоковая сумма

            // записываю на гпу массивы которые буду использовать, чтобы каждый раз не читать с цпу
            buffer1_pow2_sum_gpu.writeN(as.data(), n);
            prefix_sum_accum_gpu.writeN(as.data(), n);
            unsigned int active = n; // количество активных элементов на текущей стадии
            unsigned int stage = 0; // номер стадии (начинается с 0)
            while (active > 1) { // пока не останется один элемент
                ocl_prefix_accumulation.exec( // вычисляем префиксные суммы для текущего количества активных элементов
                    gpu::WorkSize(256, n),
                    buffer1_pow2_sum_gpu,  // откуда читаем данные
                    prefix_sum_accum_gpu, // куда пишем префиксные суммы
                    n,
                    stage
                );
                // вычисляем сумму пар элементов для следующей стадии редукции
                const unsigned int reduced = (active >> 1) + (active & 1);
                ocl_sum_reduction.exec(
                    gpu::WorkSize(256, reduced),
                    buffer1_pow2_sum_gpu,
                    buffer2_pow2_sum_gpu,
                    active
                );
                // меняем буферы местами для следующей итерации
                buffer1_pow2_sum_gpu.swap(buffer2_pow2_sum_gpu);
                active = reduced; // обновляем количество активных элементов
                ++stage; // переходим к следующей стадии
            }
        }

        times.push_back(t.elapsed());
    }
    std::cout << "prefix sum times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили префиксные суммы)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "prefix sum median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_prefix_sum = prefix_sum_accum_gpu.readVector();

    // Сверяем результат
    size_t cpu_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        cpu_sum += as[i];
        rassert(cpu_sum == gpu_prefix_sum[i], 566324523452323, cpu_sum, gpu_prefix_sum[i], i);
    }

    // Проверяем что входные данные остались нетронуты (ведь мы их переиспользуем от итерации к итерации)
    std::vector<unsigned int> input_values = input_gpu.readVector();
    for (size_t i = 0; i < n; ++i) {
        rassert(input_values[i] == as[i], 6573452432, input_values[i], as[i]);
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
