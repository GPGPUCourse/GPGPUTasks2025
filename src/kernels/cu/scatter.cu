#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void RSScatter(
    unsigned int* input,
    unsigned int* counted_buf,
    unsigned int* preffix_sums_buf0,
    unsigned int* preffix_sums_buf1,
    unsigned int* output_buf,
    unsigned int n,
    unsigned int offset
){
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    int is_zero = counted_buf[i];
    // printf("threadIdx = %d, is_zero = %d, input[i] = %d\n", threadIdx.x, is_zero, input[i]);
    if (is_zero) {
        // printf("is_zero threadIdx = %d, is_zero = %d, input[i] = %d, goes_to= %d\n",
        // threadIdx.x, is_zero, input[i], preffix_sums_buf0[i]-1);
        output_buf[preffix_sums_buf0[i]-1] = input[i];
    } else {
        //printf("NOT is_zero threadIdx = %d, is_zero = %d, input[i] = %d, goes_to = %d\n",
        //    threadIdx.x, is_zero, input[i], preffix_sums_buf1[i]+offset-1);
        output_buf[preffix_sums_buf1[i]+offset-1] = input[i];
    }
}

namespace cuda {

void RadixSortScatter(
    const gpu::WorkSize& workSize,
    gpu::gpu_mem_32u& input_buf,
    gpu::gpu_mem_32u& counted_buf,
    gpu::gpu_mem_32u& preffix_sums_buf0,
    gpu::gpu_mem_32u& preffix_sums_buf1,
    gpu::gpu_mem_32u& output_buf,
    unsigned int n,
    unsigned int offset)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::RSScatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        input_buf.cuptr(),
        counted_buf.cuptr(),
        preffix_sums_buf0.cuptr(),
        preffix_sums_buf1.cuptr(),
        output_buf.cuptr(),
        n,
        offset
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
