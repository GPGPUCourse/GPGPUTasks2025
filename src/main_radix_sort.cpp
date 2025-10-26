#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h" // TODO очень советую использовать debug::prettyBits(...) для отладки

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

    ocl::KernelSource ocl_fillBufferWithZeros(ocl::getFillBufferWithZeros());
    // обнуляю все тут
    auto zero_on_gpu = [&](gpu::gpu_mem_32u& buf, size_t count) {
        size_t l0 = 256;
        size_t g0 = ((count + l0 - 1) / l0) * l0;
        ocl_fillBufferWithZeros.exec(gpu::WorkSize(l0, g0), buf, (unsigned)count);
    };
    ocl::KernelSource ocl_radixSort01LocalCounting(ocl::getRadixSort01LocalCounting());
    ocl::KernelSource ocl_radixSort02GlobalPrefixesScanSumReduction(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    ocl::KernelSource ocl_radixSort03GlobalPrefixesScanAccumulation(ocl::getRadixSort03GlobalPrefixesScanAccumulation());
    ocl::KernelSource ocl_radixSort04Scatter(ocl::getRadixSort04Scatter());

    avk2::KernelSource vk_fillBufferWithZeros(avk2::getFillBufferWithZeros());
    avk2::KernelSource vk_radixSort01LocalCounting(avk2::getRadixSort01LocalCounting());
    avk2::KernelSource vk_radixSort02GlobalPrefixesScanSumReduction(avk2::getRadixSort02GlobalPrefixesScanSumReduction());
    avk2::KernelSource vk_radixSort03GlobalPrefixesScanAccumulation(avk2::getRadixSort03GlobalPrefixesScanAccumulation());
    avk2::KernelSource vk_radixSort04Scatter(avk2::getRadixSort04Scatter());

    FastRandom r;
    const unsigned r_bits = 8;
    const unsigned bins   = 1u << r_bits;

    int n = 100000000;
//    int max_value = 8;
    int max_value = std::numeric_limits<int>::max(); // TODO при отладке используйте минимальное max_value (например max_value=8) при котором воспроизводится бага
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> sorted(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r.next(0, max_value);
    }
    std::cout << "n=" << n << " max_value=" << max_value << std::endl;
    {
        size_t force_duplicates_attempts = 3;
        bool all_attempts_missed = true;
        for (size_t k = 0; k < force_duplicates_attempts; ++k) {
            size_t i = r.next(0, n - 1);
            size_t j = r.next(0, n - 1);
            if (i != j) {
                as[j] = as[i];
                all_attempts_missed = false;
            }
        }
        rassert(!all_attempts_missed, 4353245123412);
    }
    {
        sorted = as;
        std::cout << "sorting on CPU..." << std::endl;
        timer t;
        std::sort(sorted.begin(), sorted.end());
        double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU std::sort finished in " << t.elapsed() << " sec" << std::endl;
        std::cout << "CPU std::sort effective RAM bandwidth: " << memory_size_gb / t.elapsed() << " GB/s (" << n / 1000 / 1000 / t.elapsed() << " uint millions/s)" << std::endl;
    }
    size_t lsz = std::min(256, std::max(1, n));
    size_t gsz = ((n + (int)lsz - 1) / (int)lsz) * lsz;
    unsigned num_groups = gsz / lsz;
    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u buffer1_gpu(n);
    gpu::gpu_mem_32u buffer2_gpu(num_groups * bins);
    gpu::gpu_mem_32u buffer3_gpu(bins);
    gpu::gpu_mem_32u buffer4_gpu(num_groups * bins);
    gpu::gpu_mem_32u buffer_output_gpu(n);
    gpu::gpu_mem_32u* final_buf = &buffer_output_gpu;

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
    // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
    // Если вам нужно занулять буферы в процессе вычислений - используйте кернел который это сделает (см. кернел fill_buffer_with_zeros)
    zero_on_gpu(buffer1_gpu, n);
    zero_on_gpu(buffer2_gpu, (size_t)num_groups * bins);
    zero_on_gpu(buffer3_gpu, (size_t)bins);
//    zero_on_gpu(buffer4_gpu, (size_t)num_groups * bins);
    zero_on_gpu(buffer_output_gpu, n);

//    buffer_output_gpu.fill(255);
    std::vector<double> times;
    for (int iter = 0; iter < 20; ++iter) {
        timer t;
        if (context.type() == gpu::Context::TypeOpenCL) {
//            const unsigned r_bits = 11;
//            const unsigned bins   = 1u << r_bits;
//            size_t gsz = n;
            gpu::gpu_mem_32u* cur_in  = &input_gpu; // вход
            gpu::gpu_mem_32u* cur_out = &buffer_output_gpu; // выход
            bool first_pass = true; // флаг для первой итераци по битам
            for (unsigned bit = 0; bit < 32; bit += r_bits) { // идем по рбит
                zero_on_gpu(buffer2_gpu, (size_t)num_groups * bins); // обнуляю лок гистограммы
                zero_on_gpu(buffer3_gpu, (size_t)bins); // обнул глоб префиксы
                zero_on_gpu(buffer4_gpu, (size_t)num_groups * bins); // обнул оффсет групп
                gpu::WorkSize ws_main(lsz, gsz); // размер для кернела скаттер
                gpu::WorkSize ws_hist(lsz, gsz); // размер для гистограм
                gpu::WorkSize ws_bins(1, bins); // размер для бинов
                gpu::WorkSize ws_one(1, 1); // один поток (нужен для реалищаци)

                ocl_radixSort01LocalCounting.exec( // считается локальные гистограммы по группам
                    ws_hist, // размеры запуска
                    *cur_in, // входные данные
                    buffer2_gpu, // выходгистограммы
                    bit, // текущий сдвиг разряда
                    bins, // число корзин
                    (unsigned)n // длина массива
                );

                ocl_radixSort02GlobalPrefixesScanSumReduction.exec( // сумм по группам в общий бин
                    ws_bins, // по всем бинам
                    buffer2_gpu, // вход локальные гистограммы
                    buffer3_gpu, // выход сумма по бинам
                    bins, // число бинов
                    (unsigned)num_groups // число групп
                );

                ocl_radixSort03GlobalPrefixesScanAccumulation.exec( // префикс по бинам до тек элемента
                    ws_one, // один поток
                    buffer3_gpu, // вход суммы по бинам
                    buffer3_gpu, // выход начала бинов
                    bins, // число бинов
                    0 // a2==0 -> линейный префикс
                );

                ocl_radixSort03GlobalPrefixesScanAccumulation.exec( // префикс по группам внутри каждого бина
                    ws_bins, // по каждому бину
                    buffer2_gpu, // вход локальные гистограммы
                    buffer2_gpu, // выход локальные начала
                    bins, // число бинов
                    (unsigned)num_groups // число групп
                );

                ocl_radixSort04Scatter.exec( // расскалдывание элементов по позициям заданным
                    ws_main, // размеры запуска
                    *cur_in, // вход значения
                    buffer3_gpu, // вход глобальные начала бинов
                    *cur_out, // выход массив
                    buffer2_gpu, // вход локальные оффсеты групп
                    bit, bins, (unsigned)n // разряд и длина
                );

                if (first_pass) // если первый проход
                {
                    cur_in = cur_out; // следующий вход - полученый выход
                    cur_out = &buffer1_gpu; // буфер для следующего вывода
                    first_pass = false;
                } else
                {
                    std::swap(cur_in, cur_out);
                }

            }
            final_buf = cur_in;
        }
        times.push_back(t.elapsed());
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1000 / 1000 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = final_buf->readVector();

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(sorted[i] == gpu_sorted[i], 566324523452323, sorted[i], gpu_sorted[i], i);
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
