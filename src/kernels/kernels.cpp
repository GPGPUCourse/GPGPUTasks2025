#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/mandelbrot.h"
#include "cl/generated_kernels/sum_01_atomics.h"
#include "cl/generated_kernels/sum_02_atomics_load_k.h"
#include "cl/generated_kernels/sum_03_local_memory_atomic_per_workgroup.h"
#include "cl/generated_kernels/sum_04_local_reduction.h"

#include "vk/generated_kernels/aplusb_comp.h"
#include "vk/generated_kernels/mandelbrot_comp.h"
#include "vk/generated_kernels/sum_01_atomics_comp.h"
#include "vk/generated_kernels/sum_02_atomics_load_k_comp.h"
#include "vk/generated_kernels/sum_03_local_memory_atomic_per_workgroup_comp.h"
#include "vk/generated_kernels/sum_04_local_reduction_comp.h"

#ifndef CUDA_SUPPORT
namespace cuda {
void aplusb(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void mandelbrot(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &results,
    unsigned int width, unsigned int height,
    float fromX, float fromY,
    float sizeX, float sizeY,
    unsigned int iters, unsigned int isSmoothing)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void sum_01_atomics(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, gpu::gpu_mem_32u& sum, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 546237686412414);
}
void sum_02_atomics_load_k(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, gpu::gpu_mem_32u& sum, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 545764523412414);
}
void sum_03_local_memory_atomic_per_workgroup(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, gpu::gpu_mem_32u& sum, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 7657564523412414);
}
void sum_04_local_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, gpu::gpu_mem_32u& sum, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 7657564523412414);
}
} // namespace cuda
#endif

namespace ocl {
const ocl::ProgramBinaries& getAplusB()
{
    return opencl_binaries_aplusb;
}
const ocl::ProgramBinaries& getMandelbrot()
{
    return opencl_binaries_mandelbrot;
}
const ProgramBinaries& getSum01Atomics()
{
    return opencl_binaries_sum_01_atomics;
}
const ProgramBinaries& getSum02AtomicsLoadK()
{
    return opencl_binaries_sum_02_atomics_load_k;
}
const ProgramBinaries& getSum03LocalMemoryAtomicPerWorkgroup()
{
    return opencl_binaries_sum_03_local_memory_atomic_per_workgroup;
}
const ProgramBinaries& getSum04LocalReduction()
{
    return opencl_binaries_sum_04_local_reduction;
}
} // namespace ocl

namespace avk2 {
const ProgramBinaries& getAplusB()
{
    return vulkan_binaries_aplusb_comp;
}
const avk2::ProgramBinaries& getMandelbrot()
{
    return vulkan_binaries_mandelbrot_comp;
}
const ProgramBinaries& getSum01Atomics()
{
    return vulkan_binaries_sum_01_atomics_comp;
}
const ProgramBinaries& getSum02AtomicsLoadK()
{
    return vulkan_binaries_sum_02_atomics_load_k_comp;
}
const ProgramBinaries& getSum03LocalMemoryAtomicPerWorkgroup()
{
    return vulkan_binaries_sum_03_local_memory_atomic_per_workgroup_comp;
}
const ProgramBinaries& getSum04LocalReduction()
{
    return vulkan_binaries_sum_04_local_reduction_comp;
}
} // namespace avk2
