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
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {return;}

    int pairLen = 2 * sorted_k;
    int pairIdx = idx / pairLen;
    int pairStart = pairIdx * pairLen;
    int t = idx - pairStart;

    int leftStart = pairStart;
    int leftLen = min(sorted_k, max(0, n - leftStart));
    int rightStart = leftStart + leftLen;
    int rightLen = min(sorted_k, max(0, n - rightStart));
    int totaLen = leftLen + rightLen;

    if (t >= totaLen) {
        return;
    }

    if (rightLen == 0) {
        output_data[pairStart + t] = input_data[leftStart + t];
        return;
    }

    if (leftLen == 0) {
        output_data[pairStart + t] = input_data[rightStart + t];
        return;
    }

    int low = max(0, t - rightLen);
    int high = min(t, leftLen);
    int leftPivot = low, rightPivot = high;
    while (leftPivot < rightPivot) {
        int mid = (leftPivot + rightPivot) / 2;
        int val1 = input_data[leftStart + mid];
        int val2 = input_data[rightStart + (t - mid - 1)];
        if (val1 <= val2) {
            leftPivot = mid + 1;
        } else {
            rightPivot = mid;
        }
    }
    int a = leftPivot;
    int b = t - a;

    unsigned int result;
    if (a < leftLen && (b >= rightLen || input_data[leftStart + a] <= input_data[rightStart + b])) {
        result = input_data[leftStart + a];
    } else {
        result = input_data[rightStart + b];
    }
    output_data[pairStart + t] = result;
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
