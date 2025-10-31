#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const int next_k = sorted_k << 1;
    const int base = (i / next_k) * next_k;
    const int rem = n - base;
    const int sz = rem < next_k ? rem : next_k;
    if (sz <= sorted_k) {
        output_data[i] = input_data[i];
        return;
    }

    const unsigned int* a = input_data + base;
    const int sz1 = sorted_k;
    const unsigned int* b = input_data + base + sorted_k;
    const int sz2 = sz - sorted_k;
    const int k = i - base;

    int l = max(0, k - sz2);
    int r = min(k, sz1);
    while (l < r) {
        const int m1 = (l + r) >> 1;
        const int m2 = k - m1;
        const unsigned int val1 = (m1 > 0) ? a[m1 - 1] : 0;
        const unsigned int val2 = (m2 > 0) ? b[m2 - 1] : 0;
        if (m1 > 0 && m2 < sz2 && val1 > b[m2])
            r = m1 - 1;
        else if (m2 > 0 && m1 < sz1 && val2 > a[m1])
            l = m1 + 1;
        else {
            l = m1;
            break;
        }
    }

    const int i1 = l, i2 = k - i1;
    const unsigned int val1 = (i1 < sz1) ? a[i1] : 0xffffffffu;
    const unsigned int val2 = (i2 < sz2) ? b[i2] : 0xffffffffu;
    output_data[i] = ((val1 <= val2) ? val1 : val2);
}

namespace cuda {
void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
