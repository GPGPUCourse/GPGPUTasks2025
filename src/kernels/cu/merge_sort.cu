#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

static constexpr unsigned int BLOCK_THREADS = 256;
static constexpr unsigned int MERGE_TILE_SIZE = 1024;

__global__ void merge_sort_elementwise(
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

__global__ void merge_sort_tiled(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int i = blockIdx.x * MERGE_TILE_SIZE;
    if (i >= n)
        return;

    const unsigned int thread_ind = threadIdx.x;
    const int next_k = sorted_k << 1;
    const int base = (i / next_k) * next_k;
    const int rem = n - base;
    const int sz = rem < next_k ? rem : next_k;
    const int k = i - base;
    if (sz <= sorted_k) {
        const unsigned int end = min(MERGE_TILE_SIZE, sz > k ? sz - k : 0);
        for (unsigned int out_pos = thread_ind; out_pos < end; out_pos += BLOCK_THREADS)
            output_data[i + out_pos] = input_data[i + out_pos];
        return;
    }

    const unsigned int* a = input_data + base;
    const int sz1 = sorted_k;
    const unsigned int* b = input_data + base + sorted_k;
    const int sz2 = sz - sorted_k;

    const int global_diag_start = k;
    const int global_diag_end = min(k + MERGE_TILE_SIZE, sz);
    const unsigned int elems_cnt = (global_diag_end - global_diag_start + BLOCK_THREADS - 1) / BLOCK_THREADS;
    const int diag_start = global_diag_start + thread_ind * elems_cnt;
    if (diag_start >= global_diag_end)
        return;
    const int diag_end = min(diag_start + elems_cnt, global_diag_end);

    int l = max(0, diag_start - sz2);
    int r = min(diag_start, sz1);
    while (l < r) {
        const int m1 = (l + r) >> 1;
        const int m2 = diag_start - m1;
        unsigned int val1 = (m1 > 0) ? a[m1 - 1] : 0;
        unsigned int val2 = (m2 < sz2) ? b[m2] : 0xffffffffu;
        if (val1 > val2)
            r = m1 - 1;
        else {
            val1 = (m1 < sz1) ? a[m1] : 0xffffffffu;
            val2 = (m2 > 0) ? b[m2 - 1] : 0;
            if (val2 > val1)
                l = m1 + 1;
            else {
                l = m1;
                break;
            }
        }
    }
    int i1 = l, i2 = diag_start - l;
    for (int out_pos = diag_start; out_pos < diag_end; ++out_pos) {
        const unsigned int val1 = (i1 < sz1) ? a[i1] : 0xffffffffu;
        const unsigned int val2 = (i2 < sz2) ? b[i2] : 0xffffffffu;
        if (val1 <= val2) {
            output_data[base + out_pos] = val1;
            ++i1;
        } else {
            output_data[base + out_pos] = val2;
            ++i2;
        }
    }
}

namespace cuda {
void merge_sort(const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, const int& sorted_k, const int& n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    if ((sorted_k << 1) < MERGE_TILE_SIZE)
        ::merge_sort_elementwise<<<(n + BLOCK_THREADS - 1) / BLOCK_THREADS, BLOCK_THREADS, 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    else
        ::merge_sort_tiled<<<(n + MERGE_TILE_SIZE - 1) / MERGE_TILE_SIZE, BLOCK_THREADS, 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
