#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/aplusb_matrix_bad.h"
#include "cl/generated_kernels/aplusb_matrix_good.h"

#include "vk/generated_kernels/aplusb_comp.h"
#include "vk/generated_kernels/aplusb_matrix_bad_comp.h"
#include "vk/generated_kernels/aplusb_matrix_good_comp.h"

#ifndef CUDA_SUPPORT
namespace cuda {
void aplusb(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void aplusb_matrix_bad(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412414);
}
void aplusb_matrix_good(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412415);
}
} // namespace cuda
#endif

namespace ocl {
const ocl::ProgramBinaries& getAplusB()
{
    return opencl_binaries_aplusb;
}
const ocl::ProgramBinaries& getAplusBMatrixBad()
{
    return opencl_binaries_aplusb_matrix_bad;
}
const ocl::ProgramBinaries& getAplusBMatrixGood()
{
    return opencl_binaries_aplusb_matrix_good;
}
} // namespace ocl

namespace avk2 {
const ProgramBinaries& getAplusB()
{
    return vulkan_binaries_aplusb_comp;
}
const ProgramBinaries& getAplusBMatrixBad()
{
    return vulkan_binaries_aplusb_matrix_bad_comp;
}
const ProgramBinaries& getAplusBMatrixGood()
{
    return vulkan_binaries_aplusb_matrix_good_comp;
}
} // namespace avk2
