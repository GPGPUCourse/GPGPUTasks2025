#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/merge_sort_big.h"
#include "cl/generated_kernels/merge_sort_medium.h"
#include "cl/generated_kernels/merge_sort_small.h"

#include "vk/generated_kernels/aplusb_comp.h"
#include "vk/generated_kernels/merge_sort_comp.h"

#ifndef CUDA_SUPPORT
namespace cuda {
void aplusb(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void merge_sort(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
} // namespace cuda
#endif

namespace ocl {
const ocl::ProgramBinaries& getAplusB()
{
    return opencl_binaries_aplusb;
}

const ProgramBinaries& getMergeSortBig()
{
    return opencl_binaries_merge_sort_big;
}

const ProgramBinaries& getMergeSortMedium()
{
    return opencl_binaries_merge_sort_medium;
}

const ProgramBinaries& getMergeSortSmall()
{
    return opencl_binaries_merge_sort_small;
}
} // namespace ocl

namespace avk2 {
const ProgramBinaries& getAplusB()
{
    return vulkan_binaries_aplusb_comp;
}

const ProgramBinaries& getMergeSort()
{
    return vulkan_binaries_merge_sort_comp;
}
} // namespace avk2
