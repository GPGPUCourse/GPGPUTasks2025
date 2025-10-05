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
 
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    unsigned int benchmarkingIters = 10;

    unsigned int width = 2048;
    unsigned int height = 2048;

    unsigned int iterationsLimit = 256;
    unsigned int isSmoothing = false;

    // first trying this: 
#if 0
    float centralX = -0.789136f;
    float centralY = -0.150316f;
    float sizeX = 0.01239f;
#elif 0
    // then this:
    float centralX = -0.5f;
    float centralY = 0.0f;
    float sizeX = 2.0f;
#elif 1
    // then this:
    float centralX = -0.8631013574141035f;
    float centralY = -0.2332321736635287f;
    float sizeX = 0.000739977f;
#endif

    float sizeY = sizeX * height / width;

    image32f cpu_results;

    ocl::KernelSource ocl_mandelbrot(ocl::getMandelbrot());

    // allocating buf to vram
    gpu::gpu_mem_32f gpu_results(width * height);

    std::vector<std::string> algorithm_names = {
        "cpu",
        "cpu_w_OpenMP",
        "gpu",
    };

    for (size_t algorithm_index = 0; algorithm_index < algorithm_names.size(); ++algorithm_index) {

        const std::string& algorithm = algorithm_names[algorithm_index];
        
	std::cout << "______________________________________________________" << std::endl;
        std::cout << "evaluating algorithm #" << (algorithm_index + 1) << "/" << algorithm_names.size() << ": " << algorithm << std::endl;

        // run alg while measuring performance
        std::vector<double> times;
        image32f current_results(width, height, 1);

        int iters_count = (algorithm == "cpu") ? 1 : 10; // single-threaded CPU is too slow
        
	for (int iter = 0; iter < iters_count; ++iter) {
		
	    timer t;

            if (algorithm == "cpu") {

                cpu::mandelbrot(current_results.ptr(), width, height, centralX - sizeX / 2.0f, centralY - sizeY / 2.0f, sizeX, sizeY, iterationsLimit, isSmoothing, false);
                cpu_results = current_results;
            } else if (algorithm == "cpu_w_OpenMP") {

                if (iter == 0) std::cout << "OpenMP threads: x" << getOpenMPThreadsCount() << " threads" << std::endl;

                cpu::mandelbrot(current_results.ptr(), width, height, centralX - sizeX / 2.0f, centralY - sizeY / 2.0f, sizeX, sizeY, iterationsLimit, isSmoothing, true);
            } else if (algorithm == "gpu") {

   		if (context.type() == gpu::Context::TypeOpenCL) {

		    gpu::WorkSize work_size(GROUP_SIZE_X, GROUP_SIZE_Y, width, height);

                    ocl_mandelbrot.exec(work_size, gpu_results, width, height, centralX - sizeX / 2.0f, centralY - sizeY / 2.0f, sizeX, sizeY, iterationsLimit, isSmoothing);

		    //throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
		} else {
		    
		    rassert(false, 546345243, context.type());
                }
            }

            times.push_back(t.elapsed());

            if (algorithm == "gpu") {

                gpu_results.readN(current_results.ptr(), width * height);
            }
        }
        std::cout << "algorithm times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

	// compute max eff throughput
        size_t flopsInLoop = 10;
        size_t maxApproximateFlops = width * height * iterationsLimit * flopsInLoop;
        size_t gflops = 1000*1000*1000;
        std::cout << "mandelbrot effective algorithm gflops: " << maxApproximateFlops / gflops / stats::median(times) << " gflops" << std::endl;

        // save image
        image8u image = renderToColor(current_results.ptr(), width, height);
        std::string filename = "mandelbrot_" + algorithm + ".bmp";
        std::cout << "saving image to '" << filename << "'..." << std::endl;
        image.saveBMP(filename);

        // check
        if (!cpu_results.isNull()) {
            double errorAvg = 0.0;
            for (int j = 0; j < height; ++j) {
                for (int i = 0; i < width; ++i) {
                    errorAvg += fabs(current_results.ptr()[j * width + i] - cpu_results.ptr()[j * width + i]);
                }
            }
            errorAvg /= width * height;
            std::cout << algorithm << " vs cpu average results difference: " << 100.0 * errorAvg << "%" << std::endl;

            if (errorAvg > 0.03) {
                throw std::runtime_error("difference too high between cpu and gpu results!");
            }
        }
    }
}

int main(int argc, char** argv)
{
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {
		
	    return 0;
        }
        if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
    
	    return 0;
        } else {
    
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

            // palette from: http://iquilezles.org/www/articles/palettes/palettes.htm
            float t = results[j * width + i];
            vec3f a(0.5, 0.5, 0.5);
            vec3f b(0.5, 0.5, 0.5);
            vec3f c(1.0, 1.0, 1.0);
            vec3f d(0.00, 0.10, 0.20);
            vec3f color = a + b * cos(2*3.14f*(c*t+d));
            img_rgb[j * 3 * width + i * 3 + 0] = (unsigned char) (color.x * 255);
            img_rgb[j * 3 * width + i * 3 + 1] = (unsigned char) (color.y * 255);
            img_rgb[j * 3 * width + i * 3 + 2] = (unsigned char) (color.z * 255);
        }
    }
    return image;
}
