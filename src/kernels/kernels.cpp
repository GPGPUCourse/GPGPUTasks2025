#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/prefix_sum_01_reduction.h"
#include "cl/generated_kernels/prefix_sum_02_prefix_accumulation.h"

#include "vk/generated_kernels/aplusb_comp.h"
#include "vk/generated_kernels/fill_buffer_with_zeros_comp.h"
#include "vk/generated_kernels/prefix_sum_01_reduction_comp.h"
#include "vk/generated_kernels/prefix_sum_02_prefix_accumulation_comp.h"

#ifndef CUDA_SUPPORT
namespace cuda {
void aplusb(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void fill_buffer_with_zeros(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u &buffer, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void prefix_sum_01_sum_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &next_pow2_sum, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n, unsigned int pow2)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 546237686412414);
}
} // namespace cuda
#endif

namespace ocl {
const ocl::ProgramBinaries& getAplusB()
{
    return opencl_binaries_aplusb;
}

const ProgramBinaries& getPrefixSum01Reduction()
{
    return opencl_binaries_prefix_sum_01_reduction;
}

const ProgramBinaries& getPrefixSum02PrefixAccumulation()
{
    return opencl_binaries_prefix_sum_02_prefix_accumulation;
}
} // namespace ocl

namespace avk2 {
const ProgramBinaries& getAplusB()
{
    return vulkan_binaries_aplusb_comp;
}

const ProgramBinaries& getFillBufferWithZeros()
{
    return vulkan_binaries_fill_buffer_with_zeros_comp;
}

const ProgramBinaries& getPrefixSum01Reduction()
{
    return vulkan_binaries_prefix_sum_01_reduction_comp;
}

const ProgramBinaries& getPrefixSum02PrefixAccumulation()
{
    return vulkan_binaries_prefix_sum_02_prefix_accumulation_comp;
}
} // namespace avk2
