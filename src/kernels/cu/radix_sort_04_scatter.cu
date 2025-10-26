#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned int* arr_in,
    const unsigned int* pref_sum,
    unsigned int* arr_out,
    unsigned int n,
    unsigned int offset)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    const unsigned int elem = arr_in[idx];
    if (((elem >> offset) & 0x1) == 0) {
        const unsigned int elem_out_idx = pref_sum[idx] - 1;
        curassert(elem_out_idx > 0, 42)
        // printf("Elem #%d (%d) -> #%d\n", idx, elem, elem_out_idx);
        arr_out[elem_out_idx] = elem;
    } else {
        const unsigned int elem_out_idx = pref_sum[n - 1] + (idx - pref_sum[idx]);
        curassert(elem_out_idx < n, 42)
        // printf("Elem #%d (%d) -> #%d\n", idx, elem, elem_out_idx);
        arr_out[elem_out_idx] = elem;
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& arr_in, const gpu::gpu_mem_32u& pref_sum, gpu::gpu_mem_32u& arr_out, unsigned int n, unsigned int offset)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(arr_in.cuptr(), pref_sum.cuptr(), arr_out.cuptr(), n, offset);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda