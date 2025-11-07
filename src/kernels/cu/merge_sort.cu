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
    const int run_start = blockIdx.x * (sorted_k << 1);
    if (run_start >= n) return;

    const unsigned int* A = input_data + run_start;
    const unsigned int* B = input_data + run_start + sorted_k;

    const int lenA = min(sorted_k, n - run_start);
    const int lenB = max(0, min(sorted_k, n - (run_start + sorted_k)));
    if (lenA <= 0) return;

    unsigned int* C = output_data + run_start;

    const int bInx = blockDim.x;  
    const int tIdx = threadIdx.x;
    const int lenC = lenA + lenB;  

  
    if (lenB == 0) {
        const int chunk = (lenA + bInx - 1) / bInx;
        const int beg = tIdx * chunk;
        const int end = min(lenA, beg + chunk);
        for (int i = beg; i < end; ++i) C[i] = A[i];
        return;
    }

    for (int k = tIdx; k < lenC; k += bInx) {

        float fraction = float(k + 1) * float(lenA) / float(lenA + lenB);
        int leftIdx  = min(lenA, max(0, int(fraction)));
        int rightIdx = (k + 1) - leftIdx;

        while (leftIdx > 0 && rightIdx < lenB && A[leftIdx - 1] > B[rightIdx]) {
            leftIdx--;
            rightIdx++;
        }

        while (rightIdx > 0 && leftIdx < lenA && B[rightIdx - 1] > A[leftIdx]) {
            leftIdx++;
            rightIdx--;
        }

        unsigned int a_left = (leftIdx > 0) ? A[leftIdx - 1] : 0;
        unsigned int b_left = (rightIdx > 0) ? B[rightIdx - 1] : 0;
        C[k] = (a_left >= b_left) ? a_left : b_left;
    }
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

