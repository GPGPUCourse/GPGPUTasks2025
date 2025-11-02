#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

static constexpr unsigned int BLOCK_THREADS = 256;
static constexpr unsigned int MERGE_TILE_SIZE = 4096;
static constexpr unsigned int THREAD_ELEMS = 16;

__global__ void merge_sort_elementwise(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  n)
{
    __shared__ unsigned int smem[MERGE_TILE_SIZE << 1];
    unsigned int* src = smem;
    unsigned int* dst = smem + MERGE_TILE_SIZE;

    const unsigned int ind = threadIdx.x << 4;
    const unsigned int global_base = blockIdx.x * MERGE_TILE_SIZE + ind;

#pragma unroll
    for (unsigned int seg = 0; seg < THREAD_ELEMS; seg += 4) {
        const unsigned int cur_base = global_base + seg;
        const unsigned int cur_ind = ind + seg;
        if (cur_base + 3u < n) {
            const uint4 v = reinterpret_cast<const uint4*>(input_data + cur_base)[0];
            src[cur_ind] = v.x;
            src[cur_ind + 1u] = v.y;
            src[cur_ind + 2u] = v.z;
            src[cur_ind + 3u] = v.w;
        } else {
            src[cur_ind] = (cur_base < n) ? input_data[cur_base] : 0xffffffffu;
            src[cur_ind + 1u] = (cur_base + 1u < n) ? input_data[cur_base + 1u] : 0xffffffffu;
            src[cur_ind + 2u] = (cur_base + 2u < n) ? input_data[cur_base + 2u] : 0xffffffffu;
            src[cur_ind + 3u] = (cur_base + 3u < n) ? input_data[cur_base + 3u] : 0xffffffffu;
        }
    }
    __syncthreads();

#pragma unroll 12
    for (int sorted_k = 1; sorted_k < MERGE_TILE_SIZE; sorted_k <<= 1) {
        const int next_k = sorted_k << 1;
        const int mask = ~(next_k - 1);
#pragma unroll
        for (unsigned int seg = 0; seg < THREAD_ELEMS; seg += 4) {
            int produced = 0;
            int i = ind + seg;
            while (produced < 4) {
                const int base = i & mask;
                const int rem = MERGE_TILE_SIZE - base;
                const int sz = rem < next_k ? rem : next_k;
                const int k = i - base;
                const int can = min(4 - produced, sz - k);
                if (sz <= sorted_k) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (j < can)
                            dst[i + j] = src[i + j];
                    }
                    produced += can;
                    i += can;
                    continue;
                }

                const unsigned int* a = src + base;
                const int sz1 = sorted_k;
                const unsigned int* b = a + sorted_k;
                const int sz2 = sz - sorted_k;

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

                int i1 = l, i2 = k - i1;
                int out_pos = i;
#pragma unroll
                for (int j = 0; j < can; ++j) {
                    const unsigned int val1 = (i1 < sz1) ? a[i1] : 0xffffffffu;
                    const unsigned int val2 = (i2 < sz2) ? b[i2] : 0xffffffffu;
                    if (val1 <= val2) {
                        dst[out_pos++] = val1;
                        ++i1;
                    } else {
                        dst[out_pos++] = val2;
                        ++i2;
                    }
                }
                produced += can;
                i += can;
            }
        }
        __syncthreads();
        unsigned int* tmp = src;
        src = dst;
        dst = tmp;
    }

#pragma unroll
    for (unsigned int seg = 0; seg < THREAD_ELEMS; seg += 4) {
        const unsigned int cur_base = global_base + seg;
        const unsigned int cur_ind = ind + seg;
        if (cur_base + 3u < n)
            reinterpret_cast<uint4*>(output_data + cur_base)[0] = make_uint4(src[cur_ind], src[cur_ind + 1u], src[cur_ind + 2u], src[cur_ind + 3u]);
        else {
            if (cur_base < n)
                output_data[cur_base] = src[cur_ind];
            if (cur_base + 1u < n)
                output_data[cur_base + 1u] = src[cur_ind + 1u];
            if (cur_base + 2u < n)
                output_data[cur_base + 2u] = src[cur_ind + 2u];
            if (cur_base + 3u < n)
                output_data[cur_base + 3u] = src[cur_ind + 3u];
        }
    }
}

__global__ void merge_sort_tiled(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    __shared__ unsigned int as[MERGE_TILE_SIZE];
    __shared__ unsigned int bs[MERGE_TILE_SIZE];

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

    const int global_diag_end = min(k + MERGE_TILE_SIZE, sz);
    const int global_elems_cnt = global_diag_end - k;
    const int elems_cnt = (global_elems_cnt + BLOCK_THREADS - 1) / BLOCK_THREADS;
    const int diag_start = min(thread_ind * elems_cnt, global_elems_cnt);
    const int diag_end = min(diag_start + elems_cnt, global_elems_cnt);

    __shared__ int l1, r1, l2, r2;
    if (thread_ind == 0) {
        int l = max(0, k - sz2);
        int r = min(k, sz1);
        while (l < r) {
            const int m = (l + r) >> 1;
            if (a[m] <= b[k - m - 1])
                l = m + 1;
            else
                r = m;
        }
        l1 = l;
        r1 = k - l1;

        l = max(0, global_diag_end - sz2);
        r = min(global_diag_end, sz1);
        while (l < r) {
            const int m = (l + r) >> 1;
            if (a[m] <= b[global_diag_end - m - 1])
                l = m + 1;
            else
                r = m;
        }
        l2 = l;
        r2 = global_diag_end - l2;
    }
    __syncthreads();

    const int sub_sz1 = l2 - l1;
    const int sub_sz2 = r2 - r1;
    for (int j = thread_ind; j < sub_sz1; j += BLOCK_THREADS)
        as[j] = a[l1 + j];
    for (int j = thread_ind; j < sub_sz2; j += BLOCK_THREADS)
        bs[j] = b[r1 + j];
    __syncthreads();

    int l = max(0, diag_start - sub_sz2);
    int r = min(diag_start, sub_sz1);
    while (l < r) {
        const int m = (l + r) >> 1;
        if (as[m] <= bs[diag_start - m - 1])
            l = m + 1;
        else
            r = m;
    }
    int i1 = l, i2 = diag_start - l;
    int out_pos = base + k + diag_start;
    for (int j = diag_start; j < diag_end; ++j) {
        const unsigned int val1 = (i1 < sub_sz1) ? as[i1] : 0xffffffffu;
        const unsigned int val2 = (i2 < sub_sz2) ? bs[i2] : 0xffffffffu;
        if (val1 <= val2) {
            output_data[out_pos++] = val1;
            ++i1;
        } else {
            output_data[out_pos++] = val2;
            ++i2;
        }
    }
}

namespace cuda {
void merge_sort_elementwise(const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, const int& n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort_elementwise<<<(n + MERGE_TILE_SIZE - 1) / MERGE_TILE_SIZE, BLOCK_THREADS, 0, stream>>>(input_data.cuptr(), output_data.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}

void merge_sort_tiled(const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, const int& sorted_k, const int& n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort_tiled<<<(n + MERGE_TILE_SIZE - 1) / MERGE_TILE_SIZE, BLOCK_THREADS, 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
