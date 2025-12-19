#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libimages/debug_io.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "io/camera_reader.h"
#include "io/scene_reader.h"

#include "cpu_helpers/build_bvh_cpu.h"

#include "GL/glew.h"
#include "EGL/egl.h"

#include <cfloat>
#include <filesystem>
#include <fstream>
#include <unordered_set>

// для самого быстрого построения BVH нужно использовать операции на подгруппах
// операции на подгруппах в opencl требуют очень редкие расширения cl_khr_subgroup_*
// их нету ни в одном из двух драйверов для моего GPU, которые я нашёл
// остаётся vulkan
// но для трассировки лучей нужно использовать инструкции от amd, которых нету в spir-v*
// остаётся opencl
// вводим в гугл запрос "vulkan opencl interop" и обнаруживаем что нужно cl_khr_external_memory
// его нет у amd
// к счастью, у amd есть cl_khr_gl_sharing и GL_EXT_memory_object_fd!
// *на самом деле был вариант использовать встроенные построение/обход BVH из vulkan, но это слишком безболезненный путь.
gpu::Context opencl, vulkan;

static bool inside(AABBGPU a, AABBGPU b) {
    bool ok = true;
    ok &= b.min_x <= a.min_x && a.max_x <= b.max_x;
    ok &= b.min_y <= a.min_y && a.max_y <= b.max_y;
    ok &= b.min_z <= a.min_z && a.max_z <= b.max_z;
    return ok;
}
static bool inside(float* pt, AABBGPU b) {
    bool ok = true;
    ok &= b.min_x <= pt[0] && pt[0] <= b.max_x;
    ok &= b.min_y <= pt[1] && pt[1] <= b.max_y;
    ok &= b.min_z <= pt[2] && pt[2] <= b.max_z;
    return ok;
}
static double surface_area(AABBGPU a) {
    double x = a.max_x - a.min_x, y = a.max_y - a.min_y, z = a.max_z - a.min_z;
    return 2 * (x * y + x * z + y * z);
}

static void validate_bvh(uint8_t* data, size_t size, size_t nfaces, size_t nboxes)
{
    assert(size >= 64 * nfaces + 128 * nboxes);
    assert(nboxes <= (nfaces + 1) / 2);
    std::unordered_set<size_t> seen;
    size_t boxes = 0, dups = 0;
    printf("Verify BVH\n");
    uint64_t total_depth = 0;
    double total_sah = 0;
    auto dfs = [&](auto&& dfs, size_t at, size_t depth, AABBGPU cur_aabb) {
        if(at == -1u)
            return;
        auto[it, is_new] = seen.insert(at);
        if(!is_new) {
            dups++;
            return;
        }
        if(depth != 0)
            total_sah += surface_area(cur_aabb);
        if(at % 8 == 5) {
            boxes++;
            //printf("Box at %zu: %zu\n", depth, at);
            at = (at - 5) * 8;
            assert(at + sizeof(BVHBoxGPU) <= size);
            BVHBoxGPU* box = (BVHBoxGPU*)(data + at);
            assert(box->children[0] == -1u || inside(box->coords[0], cur_aabb));
            assert(box->children[1] == -1u || inside(box->coords[1], cur_aabb));
            assert(box->children[2] == -1u || inside(box->coords[2], cur_aabb));
            assert(box->children[3] == -1u || inside(box->coords[3], cur_aabb));
            //printf("children: %zu %zu %zu %zu\n", box->children[0], box->children[1], box->children[2], box->children[3]);
            dfs(dfs, box->children[0], depth + 1, box->coords[0]);
            dfs(dfs, box->children[1], depth + 1, box->coords[1]);
            dfs(dfs, box->children[2], depth + 1, box->coords[2]);
            dfs(dfs, box->children[3], depth + 1, box->coords[3]);
        }
        else { // triangle
            total_depth += depth;
            assert(at % 8 == 0);
            at = at * 8;
            assert(at + sizeof(BVHTriangleGPU) <= size);
            BVHTriangleGPU* tri = (BVHTriangleGPU*)(data + at);
            assert(inside(tri->a, cur_aabb));
            assert(inside(tri->b, cur_aabb));
            assert(inside(tri->c, cur_aabb));
        }
    };
    dfs(dfs, (64 * nfaces + 128 * (nboxes - 1)) / 8 + 5, 0, {-FLT_MAX,-FLT_MAX,-FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX});
    assert(boxes == nboxes);
    printf("total %zu dups %zu\n", seen.size(), dups);
    printf("mean depth %lf sah %lf\n", double(total_depth) / nfaces, total_sah / (nfaces + nboxes));
    assert(seen.size() == nboxes + nfaces);
}

static size_t buildLBVH_GPU(
                ocl::KernelSource& ocl_lbvh_prim,
                ocl::KernelSource& ocl_sort_psum,
                avk2::KernelSource& vk_sort_permute,
                avk2::KernelSource& vk_lbvh_hploc,
                unsigned nfaces,
                gpu::gpu_mem_32f& vertices,
                gpu::gpu_mem_32u& faces,
                gpu::shared_device_buffer& lbvh_vk,
                gpu::shared_device_buffer& lbvh_thin_vk,
                gpu::shared_device_buffer& lbvh_boxes_vk,
                gpu::shared_device_buffer_typed<uint32_t>& mortons_vk,
                gpu::shared_device_buffer_typed<uint32_t>& perm_vk,
                SceneGeometry& scene,
                gpu::gpu_mem_32u& hist_vk,
                gpu::shared_device_buffer_typed<uint32_t>& radix_tmp_morton,
                gpu::shared_device_buffer_typed<uint32_t>& radix_tmp_perm,
                gpu::gpu_mem_32u& radix_tmp2,
                gpu::gpu_mem_32u& parent_ids_vk,
                gpu::gpu_mem_32u& block_offsets_vk
                )
{
   //nfaces = 512;
   hist_vk.export_acquire();
   mortons_vk.export_acquire();
   lbvh_thin_vk.export_acquire();
   lbvh_vk.export_acquire();
   hist_vk.memset(0);
   auto t1 = std::chrono::steady_clock::now().time_since_epoch().count();
   ocl_lbvh_prim.exec(gpu::WorkSize(GROUP_SIZE, nfaces), nfaces, faces, vertices, lbvh_thin_vk, lbvh_vk,
                   (gpu::shared_device_buffer&)mortons_vk,
           scene.gMin.x, scene.gMin.y, scene.gMin.z,
           scene.gMax.x, scene.gMax.y, scene.gMax.z,
           hist_vk);
   // здесь мне очень хотелось просто посчитать префиксные суммы 4кб данных на процессоре
   // но как я ни менял бенчмарк ради своей выгоды, на гпу оказывалось быстрее чем дважды копировать
   // (если не считать 100мс компиляции кода на первом запуске, что эквивалентно нескольким тысячам итераций разницы)
   auto t2 = std::chrono::steady_clock::now().time_since_epoch().count();
   printf("prim: %lfms (%lfM/s)\n", (t2 - t1) / 1e6, nfaces/double(t2-t1)*1e3);
   ocl_sort_psum.exec(gpu::WorkSize(4, 4), hist_vk);

   hist_vk.export_release();
   mortons_vk.export_release();
   lbvh_thin_vk.export_release();
   lbvh_vk.export_release();
   opencl.deactivate();
   vulkan.activate();

   std::vector<uint32_t> v0;
   bool test = false;
   if(test) {
       v0 = mortons_vk.readVector();
       v0.erase(v0.begin() + nfaces, v0.end());
   }

   block_offsets_vk.memset(0);
   auto b1 = &mortons_vk, b2 = &radix_tmp_morton;
   auto b3 = &perm_vk, b4 = &radix_tmp_perm;
   for(unsigned i = 0; i < 4; i++) // 4 is even
   {
       radix_tmp2.memset(0);
       size_t per_group = 256 * SORT_DIGITS_PER_THREAD;
       struct Args { uint32_t nfaces; uint32_t i; } args;
       args.nfaces = nfaces;
       args.i = i;
       vk_sort_permute.exec(avk2::PushConstant(args), gpu::WorkSize(GROUP_SIZE, (nfaces + per_group - 1) / per_group * 256),
               (gpu::shared_device_buffer&)*b1,
               (gpu::shared_device_buffer&)*b2,
               block_offsets_vk,
               hist_vk,
               radix_tmp2,
               (gpu::shared_device_buffer&)*b3,
               (gpu::shared_device_buffer&)*b4
               );
       if(test)
       {
           auto v = b2->readVector();
           v.erase(v.begin() + nfaces, v.end());
           auto v2 = v0;
           std::stable_sort(v2.begin(), v2.end(), [i](uint64_t a, uint64_t b) {
                   if(i != 3) {
                       a %= 1ull << (8*(i+1));
                       b %= 1ull << (8*(i+1));
                   }
                   return a < b;
           });
           rassert(v.size() == v2.size(),1348728);
           bool ok = v == v2;
           printf("%u: OK: %d\n", i, ok);
           if(!ok) {
               for(size_t i = 0; i < nfaces; i++)
                   if(v[i] != v2[i])
                       printf("%zu: %zu %zu\n", i, v[i], v2[i]);
               exit(1);
           }
       }
       std::swap(b1, b2);
       std::swap(b3, b4);
   }
   auto t3 = std::chrono::steady_clock::now().time_since_epoch().count();
   printf("sort: %lfms (%lfM/s)\n", (t3 - t2) / 1e6, nfaces/double(t3-t2)*1e3);
   unsigned offsets[] { nfaces, 0};
   block_offsets_vk.write(offsets, sizeof(offsets));
   parent_ids_vk.memset(-1);
   struct Args { uint32_t nfaces; } args;
   args.nfaces = nfaces;
   vk_lbvh_hploc.exec(avk2::PushConstant(args), gpu::WorkSize(HPLOC_GROUP_SIZE, nfaces),
           mortons_vk,
           perm_vk,
           lbvh_thin_vk,
           lbvh_boxes_vk,
           lbvh_vk,
           parent_ids_vk,
           block_offsets_vk);
   auto t4 = std::chrono::steady_clock::now().time_since_epoch().count();
   printf("hploc: %lfms (%lfM/s)\n", (t4 - t3) / 1e6, nfaces/double(t4-t3)*1e3);

   block_offsets_vk.read(offsets, sizeof(offsets));
   unsigned nboxes = offsets[1];
   uint64_t root = (64 * nfaces + 128 * (nboxes - 1)) / 8 + 5;

   //printf("res: %u %u\n", offsets[0], offsets[1]);
   //std::vector<uint8_t> buf(lbvh_vk.size());
   //lbvh_vk.read(buf.data(), buf.size());
   //validate_bvh(buf.data(), buf.size(), nfaces, offsets[1]);
   //printf("%p %zu\n", buf.data(), buf.size());
   //
   //for(;;);

   vulkan.deactivate();
   opencl.activate();
   return root;
}

// Считает сколько непустых пикселей
template<typename T>
size_t countNonEmpty(const TypedImage<T> &image, T empty_value) {
    rassert(image.channels() == 1, 4523445132412, image.channels());
    size_t count = 0;
    #pragma omp parallel for reduction(+:count)
    for (ptrdiff_t j = 0; j < image.height(); ++j) {
        for (ptrdiff_t i = 0; i < image.width(); ++i) {
            if (image.ptr(j)[i] != empty_value) {
                ++count;
            }
        }
    }
    return count;
}

// Считает сколько отличающихся пикселей (отличающихся > threshold)
template<typename T>
size_t countDiffs(const TypedImage<T> &a, const TypedImage<T> &b, T threshold) {
    rassert(a.channels() == 1, 5634532413241, a.channels());
    rassert(a.channels() == b.channels(), 562435231453243);
    rassert(a.width() == b.width() && a.height() == b.height(), 562435231453243);
    size_t count = 0;
    #pragma omp parallel for reduction(+:count)
    for (ptrdiff_t j = 0; j < a.height(); ++j) {
        for (ptrdiff_t i = 0; i < a.width(); ++i) {
            if (std::abs(a.ptr(j)[i] - b.ptr(j)[i]) > threshold) {
                ++count;
            }
        }
    }
    return count;
}

void run(int argc, char** argv)
{
    EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major, minor;
    eglInitialize(eglDpy, &major, &minor);
    EGLint numConfigs;
    EGLConfig eglCfg;
    static const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };
    eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);
    eglBindAPI(EGL_OPENGL_API);
    EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);
    eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx);
    int ret = glewInit();
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    auto devs = gpu::selectAllDevices(ALL_GPUS, true);
    gpu::Device* vulkan_dev = nullptr;
    gpu::Device* opencl_dev = nullptr;
    for(auto& d : devs)
    {
        gpu::printDeviceInfo(d);
        if(d.supports_opencl && !opencl_dev)
            opencl_dev = &d;
        if(d.supports_vulkan && !vulkan_dev)
            vulkan_dev = &d;
    }

    // TODO 000 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 000 после этого изучите этот код, запустите его, изучите соответсвующий вашему выбору кернел - src/kernels/<ваш выбор>/aplusb.<ваш выбор>
    // TODO 000 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DCUDA_SUPPORT=ON в CMake options
    // TODO 010 P.S. так же в случае CUDA - добавьте в CMake options (НЕ меняйте сами CMakeLists.txt чтобы не менять окружение тестирования):
    // TODO 010 "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo" (первое - чтобы включить поддержку WMMA, второе - чтобы compute-sanitizer и профилировщик знали номера строк кернела)
    opencl = activateContext(*opencl_dev, gpu::Context::TypeOpenCL);
    printf("Started opencl\n");
    opencl.deactivate();
    vulkan = activateContext(*vulkan_dev, gpu::Context::TypeVulkan);
    printf("Started vulkan\n");
    vulkan.deactivate();
    opencl.activate();

    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_rt_brute_force(ocl::getRTBruteForce());
    ocl::KernelSource ocl_rt_with_lbvh(ocl::getRTWithLBVH());
    ocl::KernelSource ocl_rt_with_bvh4(ocl::getRTWithBVH4());
    ocl::KernelSource ocl_lbvh_prim(ocl::getLBVHPrim());
    ocl::KernelSource ocl_sort_psum(ocl::getSortPsum());
    ocl::KernelSource ocl_denoise(ocl::getDenoise());

    avk2::KernelSource vk_rt_brute_force(avk2::getRTBruteForce());
    avk2::KernelSource vk_rt_with_lbvh(avk2::getRTWithLBVH());
    avk2::KernelSource vk_sort_permute(avk2::getSortPermute());
    avk2::KernelSource vk_lbvh_hploc(avk2::getLbvhHploc());

    const std::string gnome_scene_path = "data/gnome/gnome.ply";
    std::vector<std::string> scenes = {
        gnome_scene_path,
        "data/powerplant/powerplant.obj",
        "data/san-miguel/san-miguel.obj",
    };

    const int niters = 10; // при отладке удобно запускать одну итерацию
    std::vector<double> gpu_rt_perf_mrays_per_sec;
    std::vector<double> gpu_lbvh_perfs_mtris_per_sec;

    std::cout << "Using " << AO_SAMPLES << " ray samples for ambient occlusion" << std::endl;
    for (std::string scene_path: scenes) {
        std::cout << "____________________________________________________________________________________________" << std::endl;
        timer total_t;
        if (scene_path == gnome_scene_path) {
            // data/gnome/gnome.ply содержится в репозитории, если он не нашелся - вероятно папка запуска настроена не верно
            rassert(std::filesystem::exists(scene_path), 3164718263781, "Probably wrong working directory?");
        } else if (!std::filesystem::exists(scene_path)) {
            std::cout << "Scene " << scene_path << " not found! Please download and unzip it for local evaluation - see link.txt" << std::endl;
            continue;
        }

        std::cout << "Loading scene " << scene_path << "..." << std::endl;
        timer loading_scene_t;
        SceneGeometry scene = loadScene(scene_path);
        // если на каком-то датасете падает - удобно взять подможество треугольников - например просто вызовите scene.faces.resize(10000);
        const unsigned int nvertices = scene.vertices.size();
        const unsigned int nfaces = scene.faces.size();
        rassert(nvertices > 0, 546345423523143);
        rassert(nfaces > 0, 54362452342);
        std::string scene_name = std::filesystem::path(scene_path).parent_path().filename().string();
        std::string camera_path = "data/" + scene_name + "/camera.txt";
        std::string results_dir = "results/" + scene_name;
        std::filesystem::create_directory(std::filesystem::path("results"));
        std::filesystem::create_directory(std::filesystem::path(results_dir));
        std::cout << "Loading camera " << camera_path << "..." << std::endl;
        CameraViewGPU camera = loadViewState(camera_path);
        const unsigned int width = camera.K.width;
        const unsigned int height = camera.K.height;
        double loading_data_time = loading_scene_t.elapsed();
        double images_saving_time = 0.0;
        std::cout << "Scene " << scene_name << " loaded: " << nvertices << " vertices, " << nfaces << " faces in " << loading_data_time << " sec" << std::endl;
        std::cout << "Camera framebuffer size: " << width << "x" << height << std::endl;

        // Аллоцируем буферы в VRAM
        gpu::gpu_mem_32f vertices_gpu(3 * nvertices);
        gpu::gpu_mem_32u faces_gpu(3 * nfaces);
        gpu::shared_device_buffer_typed<CameraViewGPU> camera_gpu(1);

        // Аллоцируем фрейм-буферы (то есть картинки в которые сохранится результат рендеринга)
        gpu::gpu_mem_32i framebuffer_face_id_gpu(width * height);
        gpu::gpu_mem_32f denoised_gpu(width * height);
        gpu::gpu_mem_32f denoised_variance_in_gpu(width * height);
        gpu::gpu_mem_32f denoised_variance_out_gpu(width * height);
        gpu::gpu_mem_32f framebuffer_ambient_occlusion_gpu(width * height);

        // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
        timer pcie_writing_t;
        vertices_gpu.writeN((const float*) scene.vertices.data(), 3 * nvertices);
        faces_gpu.writeN((const unsigned int*) scene.faces.data(), 3 * nfaces);
        camera_gpu.writeN(&camera, 1);
        double pcie_writing_time = pcie_writing_t.elapsed();
        double pcie_reading_time = 0.0;

        // Перед каждой отрисовкой мы будем зачищать результирующие framebuffers этими значениями
        const int NO_FACE_ID = -1;
        const float NO_AMBIENT_OCCLUSION = -1.0f;
        double cleaning_framebuffers_time = 0.0;

        double brute_force_total_time = 0.0;
        image32i brute_force_framebuffer_face_ids;
        image32f brute_force_framebuffer_ambient_occlusion;
        const bool has_brute_force = (nfaces < 1000);
        
        if (has_brute_force) {
            std::vector<double> brute_force_times;
            for (int iter = 0; iter < niters; ++iter) {
                timer t;

                ocl_rt_brute_force.exec(
                    gpu::WorkSize(16, 16, width, height),
                    vertices_gpu, faces_gpu,
                    framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu,
                    camera_gpu.clmem(), nfaces);

                brute_force_times.push_back(t.elapsed());
            }
            brute_force_total_time = stats::sum(brute_force_times);
            std::cout << "GPU brute force ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(brute_force_times) << std::endl;

            // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
            timer pcie_reading_t;
            brute_force_framebuffer_face_ids = image32i(width, height, 1);
            brute_force_framebuffer_ambient_occlusion = image32f(width, height, 1);
            framebuffer_face_id_gpu.readN(brute_force_framebuffer_face_ids.ptr(), width * height);
            framebuffer_ambient_occlusion_gpu.readN(brute_force_framebuffer_ambient_occlusion.ptr(), width * height);
            pcie_reading_time += pcie_reading_t.elapsed();

            size_t non_empty_brute_force_face_ids = countNonEmpty(brute_force_framebuffer_face_ids, NO_FACE_ID);
            size_t non_empty_brute_force_ambient_occlusion = countNonEmpty(brute_force_framebuffer_ambient_occlusion, NO_AMBIENT_OCCLUSION);
            rassert(non_empty_brute_force_face_ids > width * height / 10, 2345123412, non_empty_brute_force_face_ids);
            rassert(non_empty_brute_force_ambient_occlusion > width * height / 10, 3423413421, non_empty_brute_force_face_ids);
            timer images_saving_t;
            debug_io::dumpImage(results_dir + "/framebuffer_face_ids_brute_force.bmp", debug_io::randomMapping(brute_force_framebuffer_face_ids, NO_FACE_ID));
            debug_io::dumpImage(results_dir + "/framebuffer_ambient_occlusion_brute_force.bmp", debug_io::depthMapping(brute_force_framebuffer_ambient_occlusion));
            images_saving_time += images_saving_t.elapsed();
        }

        
        double cpu_lbvh_time = 0.0;
        double rt_times_with_cpu_lbvh_sum = 0.0;
        
        {
            std::vector<BVHNodeGPU> lbvh_nodes_cpu;
            std::vector<uint32_t> leaf_faces_indices_cpu;
            timer cpu_lbvh_t;
            buildLBVH_CPU(scene.vertices, scene.faces, lbvh_nodes_cpu, leaf_faces_indices_cpu);
            cpu_lbvh_time = cpu_lbvh_t.elapsed();
            double build_mtris_per_sec = nfaces * 1e-6f / cpu_lbvh_time;
            std::cout << "CPU build LBVH in " << cpu_lbvh_time << " sec" << std::endl;
            std::cout << "CPU LBVH build performance: " << build_mtris_per_sec << " MTris/s" << std::endl;

            gpu::shared_device_buffer_typed<BVHNodeGPU> lbvh_nodes_gpu(lbvh_nodes_cpu.size());
            gpu::gpu_mem_32u leaf_faces_indices_gpu(leaf_faces_indices_cpu.size());
            lbvh_nodes_gpu.writeN(lbvh_nodes_cpu.data(), lbvh_nodes_cpu.size());
            leaf_faces_indices_gpu.writeN(leaf_faces_indices_cpu.data(), leaf_faces_indices_cpu.size());

            timer cleaning_framebuffers_t;
            framebuffer_face_id_gpu.fill(NO_FACE_ID);
            framebuffer_ambient_occlusion_gpu.fill(NO_AMBIENT_OCCLUSION);
            cleaning_framebuffers_time += cleaning_framebuffers_t.elapsed();

            std::vector<double> rt_times_with_cpu_lbvh;
            for (int iter = 0; iter < 1; ++iter) {
                timer t;

                ocl_rt_with_lbvh.exec(
                    gpu::WorkSize(16, 16, width, height),
                    vertices_gpu, faces_gpu,
                    lbvh_nodes_gpu.clmem(), leaf_faces_indices_gpu.clmem(),
                    framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu,
                    camera_gpu.clmem(), nfaces);

                rt_times_with_cpu_lbvh.push_back(t.elapsed());
            }
            rt_times_with_cpu_lbvh_sum = stats::sum(rt_times_with_cpu_lbvh);
            double mrays_per_sec = width * height * AO_SAMPLES * 1e-6f / stats::median(rt_times_with_cpu_lbvh);
            std::cout << "GPU with CPU LBVH ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(rt_times_with_cpu_lbvh) << std::endl;
            std::cout << "GPU with CPU LBVH ray tracing performance: " << mrays_per_sec << " MRays/s" << std::endl;
            //gpu_rt_perf_mrays_per_sec.push_back(mrays_per_sec);

            timer pcie_reading_t;
            image32i cpu_lbvh_framebuffer_face_ids(width, height, 1);
            image32f cpu_lbvh_framebuffer_ambient_occlusion(width, height, 1);
            framebuffer_face_id_gpu.readN(cpu_lbvh_framebuffer_face_ids.ptr(), width * height);
            framebuffer_ambient_occlusion_gpu.readN(cpu_lbvh_framebuffer_ambient_occlusion.ptr(), width * height);
            pcie_reading_time += pcie_reading_t.elapsed();

            timer cpu_lbvh_images_saving_t;
            debug_io::dumpImage(results_dir + "/framebuffer_face_ids_with_cpu_lbvh.bmp", debug_io::randomMapping(cpu_lbvh_framebuffer_face_ids, NO_FACE_ID));
            debug_io::dumpImage(results_dir + "/framebuffer_ambient_occlusion_with_cpu_lbvh.bmp", debug_io::depthMapping(cpu_lbvh_framebuffer_ambient_occlusion));
            images_saving_time += cpu_lbvh_images_saving_t.elapsed();
            if (has_brute_force) {
                unsigned int count_ao_errors = countDiffs(brute_force_framebuffer_ambient_occlusion, cpu_lbvh_framebuffer_ambient_occlusion, 0.01f);
                unsigned int count_face_id_errors = countDiffs(brute_force_framebuffer_face_ids, cpu_lbvh_framebuffer_face_ids, 1);
                rassert(count_ao_errors < width * height / 100, 345341512354123, count_ao_errors, to_percent(count_ao_errors, width * height));
                rassert(count_face_id_errors < width * height / 100, 3453415123546587, count_face_id_errors, to_percent(count_face_id_errors, width * height));
            }
        }

        
        double gpu_lbvh_time_sum = 0.0;
        double rt_times_with_gpu_lbvh_sum = 0.0;

        // TODO постройте LBVH на GPU
        // TODO оттрасируйте лучи на GPU используя построенный на GPU LBVH
        bool gpu_lbvg_gpu_rt_done = true;

        if (gpu_lbvg_gpu_rt_done) {
            opencl.deactivate();
            vulkan.activate();
            gpu::shared_device_buffer_typed<uint32_t> mortons_vk(nfaces, true);
            gpu::shared_device_buffer_typed<uint32_t> perm_vk(nfaces + 32, true);
            perm_vk.memset(-1); // pad with -1s for h-ploc OOB reads
            gpu::shared_device_buffer_typed<uint32_t> radix_tmp_morton(nfaces);
            gpu::shared_device_buffer_typed<uint32_t> radix_tmp_perm(nfaces);
            gpu::gpu_mem_32u radix_tmp2((nfaces + 255) / 256 * 256);
            gpu::gpu_mem_32u hist_vk(4 * 256, true);
            gpu::gpu_mem_32u block_offsets_vk(4);
            gpu::shared_device_buffer lbvh_thin_vk(nfaces * 2 * sizeof(BVHThinNodeGPU), true); // 2x size to handle h-ploc temporary nodes as well as input triangles
            gpu::shared_device_buffer lbvh_vk(nfaces * sizeof(BVHTriangleGPU)
                    + (nfaces + 10) / 2 * sizeof(BVHBoxGPU), true) // worst case x/3 + x/9 + ... = 1/2 (times 2 because box nodes are twice as big)
                ;
            gpu::shared_device_buffer lbvh_boxes_vk(lbvh_vk, nfaces * sizeof(BVHTriangleGPU));
            gpu::gpu_mem_32u parent_ids_vk(nfaces);
            vulkan.deactivate();
            opencl.activate();
            mortons_vk.opencl_import();
            hist_vk.opencl_import();
            lbvh_thin_vk.opencl_import();
            lbvh_vk.opencl_import();
            std::vector<double> gpu_lbvh_times;
            size_t root_ptr = 0;
            for (int iter = 0; iter < niters; ++iter) {
                timer t;

                // TODO постройте LBVH на GPU
                root_ptr = buildLBVH_GPU(ocl_lbvh_prim, ocl_sort_psum, vk_sort_permute, vk_lbvh_hploc,
                                nfaces, vertices_gpu, faces_gpu, lbvh_vk, lbvh_thin_vk, lbvh_boxes_vk, mortons_vk,
                                perm_vk, scene,
                                hist_vk, radix_tmp_morton, radix_tmp_perm, radix_tmp2,
                                parent_ids_vk, block_offsets_vk);

                gpu_lbvh_times.push_back(t.elapsed());
            }
            printf("Exited gpu BVH build loop\n");
            
            gpu_lbvh_time_sum = stats::sum(gpu_lbvh_times);
            double build_mtris_per_sec = nfaces * 1e-6f / stats::median(gpu_lbvh_times);
            std::cout << "GPU LBVH build times (in seconds) - " << stats::valuesStatsLine(gpu_lbvh_times) << std::endl;
            std::cout << "GPU LBVH build performance: " << build_mtris_per_sec << " MTris/s" << std::endl;
            gpu_lbvh_perfs_mtris_per_sec.push_back(build_mtris_per_sec);

            timer cleaning_framebuffers_t;
            framebuffer_face_id_gpu.fill(NO_FACE_ID);
            framebuffer_ambient_occlusion_gpu.fill(NO_AMBIENT_OCCLUSION);
            cleaning_framebuffers_time += cleaning_framebuffers_t.elapsed();

            std::vector<double> gpu_lbvh_rt_times;
            for (int iter = 0; iter < niters; ++iter) {
                timer t;

                // TODO оттрасируйте лучи на GPU используя построенный на GPU LBVH
                ocl_rt_with_bvh4.exec(
                    gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, width, height), lbvh_vk.clmem(),
                    framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu,
                    camera_gpu.clmem(), (unsigned)root_ptr);

                gpu_lbvh_rt_times.push_back(t.elapsed());
            }
            rt_times_with_gpu_lbvh_sum = stats::sum(gpu_lbvh_rt_times);
            double mrays_per_sec = width * height * AO_SAMPLES * 1e-6f / stats::median(gpu_lbvh_rt_times);
            std::cout << "GPU with GPU LBVH ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(gpu_lbvh_rt_times) << std::endl;
            std::cout << "GPU with GPU LBVH ray tracing performance: " << mrays_per_sec << " MRays/s" << std::endl;
            gpu_rt_perf_mrays_per_sec.push_back(mrays_per_sec);

            timer pcie_reading_t;
            image32i gpu_lbvh_framebuffer_face_ids(width, height, 1);
            image32f gpu_lbvh_framebuffer_ambient_occlusion(width, height, 1);
            framebuffer_face_id_gpu.readN(gpu_lbvh_framebuffer_face_ids.ptr(), width * height);
            framebuffer_ambient_occlusion_gpu.readN(gpu_lbvh_framebuffer_ambient_occlusion.ptr(), width * height);
            pcie_reading_time += pcie_reading_t.elapsed();

            timer gpu_lbvh_images_saving_t;
            debug_io::dumpImage(results_dir + "/framebuffer_face_ids_with_gpu_lbvh.bmp", debug_io::randomMapping(gpu_lbvh_framebuffer_face_ids, NO_FACE_ID));
            debug_io::dumpImage(results_dir + "/framebuffer_ambient_occlusion_with_gpu_lbvh.bmp", debug_io::depthMapping(gpu_lbvh_framebuffer_ambient_occlusion));
            images_saving_time += gpu_lbvh_images_saving_t.elapsed();
            if (has_brute_force) {
                unsigned int count_ao_errors = countDiffs(brute_force_framebuffer_ambient_occlusion, gpu_lbvh_framebuffer_ambient_occlusion, 0.01f);
                unsigned int count_face_id_errors = countDiffs(brute_force_framebuffer_face_ids, gpu_lbvh_framebuffer_face_ids, 1);
                rassert(count_ao_errors < width * height / 100, 3567856512354123, count_ao_errors, to_percent(count_ao_errors, width * height));
                rassert(count_face_id_errors < width * height / 100, 3453465346387, count_face_id_errors, to_percent(count_face_id_errors, width * height));
            }

            ocl_denoise.exec(gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, width, height),
                    width, height, framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu, denoised_gpu, denoised_variance_out_gpu, denoised_variance_in_gpu, -1);
            ocl_denoise.exec(gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, width, height),
                    width, height, framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu, denoised_gpu, denoised_variance_in_gpu, denoised_variance_out_gpu, 0);
            ocl_denoise.exec(gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, width, height),
                    width, height, framebuffer_face_id_gpu, denoised_gpu, framebuffer_ambient_occlusion_gpu, denoised_variance_out_gpu, denoised_variance_in_gpu, 1);
            ocl_denoise.exec(gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, width, height),
                    width, height, framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu, denoised_gpu, denoised_variance_in_gpu, denoised_variance_out_gpu, 2);
            ocl_denoise.exec(gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, width, height),
                    width, height, framebuffer_face_id_gpu, denoised_gpu, framebuffer_ambient_occlusion_gpu, denoised_variance_out_gpu, denoised_variance_in_gpu, 3);
            ocl_denoise.exec(gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, width, height),
                    width, height, framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu, denoised_gpu, denoised_variance_in_gpu, denoised_variance_out_gpu, 4);
            denoised_gpu.readN(gpu_lbvh_framebuffer_ambient_occlusion.ptr(), width * height);
            debug_io::dumpImage(results_dir + "/framebuffer_ambient_occlusion_denoised.bmp", debug_io::depthMapping(gpu_lbvh_framebuffer_ambient_occlusion));
            
            opencl.deactivate();
            vulkan.activate(); // for destructors
        }
        vulkan.deactivate();
        opencl.activate();

        double total_time = total_t.elapsed();
        std::cout << "Scene processed in " << total_t.elapsed() << " sec = ";
        std::cout << to_percent(loading_data_time, total_time) << " scene IO + ";
        if (has_brute_force) {
            std::cout << to_percent(brute_force_total_time, total_time) << " brute force RT + ";
        }
        std::cout << to_percent(cpu_lbvh_time, total_time) << " CPU LBVH + ";
        std::cout << to_percent(rt_times_with_cpu_lbvh_sum, total_time) << " GPU with CPU LBVH + ";
        std::cout << to_percent(gpu_lbvh_time_sum, total_time) << " GPU LBVH + ";
        std::cout << to_percent(rt_times_with_gpu_lbvh_sum, total_time) << " GPU with GPU LBVH + ";
        std::cout << to_percent(images_saving_time, total_time) << " images IO + ";
        std::cout << to_percent(pcie_writing_time, total_time) << " PCI-E write + ";
        std::cout << to_percent(pcie_reading_time, total_time) << " PCI-E read + ";
        std::cout << to_percent(cleaning_framebuffers_time, total_time) << " cleaning VRAM";
        std::cout << std::endl;
    }

    std::cout << "____________________________________________________________________________________________" << std::endl;
    double avg_gpu_rt_perf = stats::avg(gpu_rt_perf_mrays_per_sec);
    double avg_lbvh_build_perf = stats::avg(gpu_lbvh_perfs_mtris_per_sec);
    std::cout << "Total GPU RT with  LBVH avg perf: " << avg_gpu_rt_perf << " MRays/sec (all " << stats::vectorToString(gpu_rt_perf_mrays_per_sec) << ")" << std::endl;
    std::cout << "Total building GPU LBVH avg perf: " << avg_lbvh_build_perf << " MTris/sec (all " << stats::vectorToString(gpu_lbvh_perfs_mtris_per_sec) << ")" << std::endl;
    std::cout << "Final score: " << avg_gpu_rt_perf * avg_lbvh_build_perf << " coolness" << std::endl;
    if (gpu_rt_perf_mrays_per_sec.size() != 6 || gpu_lbvh_perfs_mtris_per_sec.size() != 3) {
        std::cout << "Results are incomplete!" << std::endl;
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
