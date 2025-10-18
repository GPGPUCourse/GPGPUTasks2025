#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define PLUS_INFINITY  UINT_MAX
#define MINUS_INFINITY 0

#define FETCH_VALUE_OR_INFINITY(buffer, i, n)  ((i < n) ? (buffer[i]) : (PLUS_INFINITY))
#define FETCH_VALUE_OR_INFINITY2(buffer, i, from, to) ((i >= from && i < to) ? (buffer[i]) : (i < from ? MINUS_INFINITY : PLUS_INFINITY))

__global__ void merge_sort(
    const unsigned int* input_data,  // size of each  input sorted chunk - sorted_k
          unsigned int* output_data, // size of each merged sorted chunk - 2 * sorted_k
                   int  sorted_k,
                   int  n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        //  input data: [0 ... sorted_k-1] [sorted_k ... 2*sorted_k-1] [2*sorted_k ... 3*sorted_k-1] [3*sorted_k ... 4*sorted_k-1] ...
        // output data: [0               ...             2*sorted_k-1] [2*sorted_k                 ...               4*sorted_k-1] ...
        //                          ^
        //                          |
        //                         input_data[i]
        //              [        chunk A ] [ chunk B                 ]
        //              [           merged chunk                     ]
        const unsigned int thread_value = input_data[i];

        const int merged_chunk = i / (2 * sorted_k);
        const int merged_chunk_from = merged_chunk * (2 * sorted_k);
        curassert(i >= merged_chunk_from && i < merged_chunk_from + 2 * sorted_k, 546231324);

        const int chunk_a_from = (2 * merged_chunk + 0) * sorted_k;
        const int chunk_a_to   = (2 * merged_chunk + 1) * sorted_k;
        const int chunk_b_from = (2 * merged_chunk + 1) * sorted_k;
        const int chunk_b_to   = min((2 * merged_chunk + 2) * sorted_k, n);

        int thread_chunk_from, thread_chunk_to, that_chunk_from, that_chunk_to;
        if (i >= chunk_a_from && i < chunk_a_to) {
            thread_chunk_from = chunk_a_from; thread_chunk_to = chunk_a_to; that_chunk_from = chunk_b_from; that_chunk_to = chunk_b_to;
        } else {
            curassert(i >= chunk_b_from && i < chunk_b_to, 546423141);
            thread_chunk_from = chunk_b_from; thread_chunk_to = chunk_b_to; that_chunk_from = chunk_a_from; that_chunk_to = chunk_a_to;
        }

        int merged_chunk_thread_position;
        if (chunk_b_from >= n) {
            merged_chunk_thread_position = i - thread_chunk_from;
        } else {
        // We want to find how many elements in that another chunk will be in output before our thread_value.
        // When we have equal values in chunk A and chunk B we suppose to put firstly elements from A, and then - elements from B,
        // so  if thread_value is from chunk A - we want to find how many elements in B <  thread_value (leftside binary search),
        // and if thread_value is from chunk B - we want to find how many elements in A <= thread_value (rigthside binary search).
        bool leftside_binary_search = (thread_chunk_from == chunk_a_from);
        int l = that_chunk_from - 1;
        int r = min(that_chunk_to, n);
        // See https://neerc.ifmo.ru/wiki/index.php?title=%D0%A6%D0%B5%D0%BB%D0%BE%D1%87%D0%B8%D1%81%D0%BB%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9_%D0%B4%D0%B2%D0%BE%D0%B8%D1%87%D0%BD%D1%8B%D0%B9_%D0%BF%D0%BE%D0%B8%D1%81%D0%BA
        while (l < r - 1) {
            int m = (l + r) / 2;
            unsigned int that_value = FETCH_VALUE_OR_INFINITY(input_data, m, n);
            if (leftside_binary_search) { if (that_value <  thread_value) { l = m; } else { r = m; } }
            else                        { if (that_value <= thread_value) { l = m; } else { r = m; } }
            if (leftside_binary_search) { curassert(FETCH_VALUE_OR_INFINITY2(input_data, r, that_chunk_from, that_chunk_to) >= thread_value, 12342546); curassert(FETCH_VALUE_OR_INFINITY2(input_data, l, that_chunk_from, that_chunk_to) < thread_value, 65443211); }
        }
        int that_chunk_elements_count_before_thread_value;
        if (leftside_binary_search) { that_chunk_elements_count_before_thread_value = r - that_chunk_from;     curassert(FETCH_VALUE_OR_INFINITY2(input_data, r, that_chunk_from, that_chunk_to) >= thread_value, 362342342); curassert(FETCH_VALUE_OR_INFINITY2(input_data, r - 1, that_chunk_from, that_chunk_to) < thread_value, 454341321); }
        else                        { that_chunk_elements_count_before_thread_value = l + 1 - that_chunk_from; curassert(FETCH_VALUE_OR_INFINITY2(input_data, l, that_chunk_from, that_chunk_to) <= thread_value, 362342343); curassert(FETCH_VALUE_OR_INFINITY2(input_data, l + 1, that_chunk_from, that_chunk_to) > thread_value, 454341322); }

        merged_chunk_thread_position = (i - thread_chunk_from) + that_chunk_elements_count_before_thread_value;
        curassert(merged_chunk_from + merged_chunk_thread_position < n, 433253112);
        }

        output_data[merged_chunk_from + merged_chunk_thread_position] = thread_value;
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
