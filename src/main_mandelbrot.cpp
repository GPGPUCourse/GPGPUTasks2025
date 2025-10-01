#include <libbase/stats.h>
#include <libbase/timer.h>
#include <libutils/misc.h>
#include <libimages/images.h>
#include <libbase/omp_utils.h>
#include <libgpu/vulkan/engine.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>

void cpu::mandelbrot(float* results,
                   unsigned int width, unsigned int height,
                   float fromX, float fromY,
                   float sizeX, float sizeY,
                   unsigned int iters, unsigned int isSmoothing,
                   bool useOpenMP)
{
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    #pragma omp parallel for if(useOpenMP)
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            float x0 = fromX + (i + 0.5f) * sizeX / width;
            float y0 = fromY + (j + 0.5f) * sizeY / height;

            float x = x0;
            float y = y0;

            int iter = 0;
            for (; iter < iters; ++iter) {
                float xPrev = x;
                x = x * x - y * y + x0;
                y = 2.0f * xPrev * y + y0;
                if ((x * x + y * y) > threshold2) {
                    break;
                }
            }
            float result = iter;
            if (isSmoothing && iter != iters) {
                result = result - logf(logf(sqrtf(x * x + y * y)) / logf(threshold)) / logf(2.0f);
            }

            result = 1.0f * result / iters;
            results[j * width + i] = result;
        }
    }
}

image8u renderToColor(const float* results, unsigned int width, unsigned int height);

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    // TODO 000 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 000 после этого изучите этот код, запустите его, изучите соответсвующий вашему выбору кернел - src/kernels/<ваш выбор>/aplusb.<ваш выбор>
    // TODO 000 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DCUDA_SUPPORT=ON в CMake options
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    unsigned int benchmarkingIters = 10;

    unsigned int width = 2048;
    unsigned int height = 2048;
    unsigned int iterationsLimit = 256;
    unsigned int isSmoothing = false;

#if 1
    float centralX = -0.789136f;
    float centralY = -0.150316f;
    float sizeX = 0.00239f;
#else
    // Менее красивый ракурс, но в этом ракурсе виден весь фрактал:
    float centralX = -0.5f;
    float centralY = 0.0f;
    float sizeX = 2.0f;
#endif
    float sizeY = sizeX * height / width;

    image32f cpu_results;

    ocl::KernelSource ocl_mandelbrot(ocl::getMandelbrot());

    avk2::KernelSource vk_mandelbrot(avk2::getMandelbrot());

    // Аллоцируем буфер в VRAM
    gpu::gpu_mem_32f gpu_results(width * height);

    std::vector<std::string> algorithm_names = {
        "CPU",
        "CPU with OpenMP",
        "GPU",
    };

    for (size_t algorithm_index = 0; algorithm_index < algorithm_names.size(); ++algorithm_index) {
        const std::string& algorithm = algorithm_names[algorithm_index];
        std::cout << "______________________________________________________" << std::endl;
        std::cout << "Evaluating algorithm #" << (algorithm_index + 1) << "/" << algorithm_names.size() << ": " << algorithm << std::endl;

        // Запускаем алгоритм (несколько раз и с замером времени выполнения)
        std::vector<double> times;
        image32f current_results(width, height, 1);
        int iters_count = (algorithm == "CPU") ? 1 : 10; // single-threaded CPU is too slow
        for (int iter = 0; iter < iters_count; ++iter) {
            timer t;

            if (algorithm == "CPU") {
                cpu::mandelbrot(current_results.ptr(), width, height, centralX - sizeX / 2.0f, centralY - sizeY / 2.0f, sizeX, sizeY, iterationsLimit, isSmoothing, false);
                cpu_results = current_results;
            } else if (algorithm == "CPU with OpenMP") {
                if (iter == 0) std::cout << "OpenMP threads: x" << getOpenMPThreadsCount() << " threads" << std::endl;
                cpu::mandelbrot(current_results.ptr(), width, height, centralX - sizeX / 2.0f, centralY - sizeY / 2.0f, sizeX, sizeY, iterationsLimit, isSmoothing, true);
            } else if (algorithm == "GPU") {
                // _______________________________OpenCL_____________________________________________
                if (context.type() == gpu::Context::TypeOpenCL) {
                    // TODO ocl_mandelbrot.exec(...);
                    // throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
                    gpu::WorkSize workSize(GROUP_SIZE_X, GROUP_SIZE_Y, width, height);
                    ocl_mandelbrot.exec(workSize, gpu_results, width, height, centralX - sizeX / 2.0f, centralY - sizeY / 2.0f, sizeX, sizeY, iterationsLimit, isSmoothing);

                    // _______________________________CUDA___________________________________________
                } else if (context.type() == gpu::Context::TypeCUDA) {
                    // TODO cuda::mandelbrot(..);
                    throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);

                    // _______________________________Vulkan_________________________________________
                } else if (context.type() == gpu::Context::TypeVulkan) {
                    typedef unsigned int uint;
                    struct {
                        uint width; uint height;
                       float fromX; float fromY;
                       float sizeX; float sizeY;
                        uint iters; uint isSmoothing;
                    } params = { width, height, centralX - sizeX / 2.0f, centralY - sizeY / 2.0f, sizeX, sizeY, iterationsLimit, isSmoothing };
                    // TODO vk_mandelbrot.exec(params, ...);
                    throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
                } else {
                    rassert(false, 546345243, context.type());
                }
            }

            times.push_back(t.elapsed());

            if (algorithm == "GPU") {
                gpu_results.readN(current_results.ptr(), width * height);
            }
        }
        std::cout << "algorithm times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        // Вычисляем достигнутую эффективную пропускную способность алгоритма (из соображений что мы все итерации делались полностью, без быстрого выхода через break)
        size_t flopsInLoop = 10;
        size_t maxApproximateFlops = width * height * iterationsLimit * flopsInLoop;
        size_t gflops = 1000*1000*1000;
        std::cout << "Mandelbrot effective algorithm GFlops: " << maxApproximateFlops / gflops / stats::median(times) << " GFlops" << std::endl;

        // Сохраняем картинку
        image8u image = renderToColor(cpu_results.ptr(), width, height);
        std::string filename = "mandelbrot " + algorithm + ".bmp";
        std::cout << "saving image to '" << filename << "'..." << std::endl;
        image.saveBMP(filename);

        // Сверяем результат
        if (!cpu_results.isNull()) {
            double errorAvg = 0.0;
            for (int j = 0; j < height; ++j) {
                for (int i = 0; i < width; ++i) {
                    errorAvg += fabs(current_results.ptr()[j * width + i] - cpu_results.ptr()[j * width + i]);
                }
            }
            errorAvg /= width * height;
            std::cout << algorithm << " vs CPU average results difference: " << 100.0 * errorAvg << "%" << std::endl;

            if (errorAvg > 0.03) {
                throw std::runtime_error("Too high difference between CPU and GPU results!");
            }
        }
    }

    // renderInWindow(centralX, centralY, iterationsLimit);
    // если захотите сделать интерактивное красивое - скажите, я дам вам заготовку которую будет несложно доделать
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

struct vec3f {
    vec3f(float x, float y, float z) : x(x), y(y), z(z) {}

    float x; float y; float z;
};

vec3f operator+(const vec3f &a, const vec3f &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

vec3f operator*(const vec3f &a, const vec3f &b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

vec3f operator*(const vec3f &a, float t) {
    return {a.x * t, a.y * t, a.z * t};
}

vec3f operator*(float t, const vec3f &a) {
    return a * t;
}

vec3f sin(const vec3f &a) {
    return {sinf(a.x), sinf(a.y), sinf(a.z)};
}

vec3f cos(const vec3f &a) {
    return {cosf(a.x), cosf(a.y), cosf(a.z)};
}

image8u renderToColor(const float* results, unsigned int width, unsigned int height)
{
    image8u image(width, height, 3);
    unsigned char *img_rgb = image.ptr();
    #pragma omp parallel for
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            // Палитра взята отсюда: http://iquilezles.org/www/articles/palettes/palettes.htm
            float t = results[j * width + i];
            vec3f a(0.5, 0.5, 0.5);
            vec3f b(0.5, 0.5, 0.5);
            vec3f c(1.0, 0.7, 0.4);
            vec3f d(0.00, 0.15, 0.20);
            vec3f color = a + b * cos(2*3.14f*(c*t+d));
            img_rgb[j * 3 * width + i * 3 + 0] = (unsigned char) (color.x * 255);
            img_rgb[j * 3 * width + i * 3 + 1] = (unsigned char) (color.y * 255);
            img_rgb[j * 3 * width + i * 3 + 2] = (unsigned char) (color.z * 255);
        }
    }
    return image;
}
