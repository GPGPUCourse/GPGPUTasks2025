#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
#include <mma.h>
using namespace nvcuda;

__global__ void matrix_multiply_wmma(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    #define WMMA_M 16
    #define WMMA_N 16
    #define WMMA_K 16

    int out_tile_row = blockIdx.y * WMMA_M;
    int out_tile_col = blockIdx.x * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    __shared__ half tile_a[WMMA_M][WMMA_K];
    __shared__ half tile_b[WMMA_K][WMMA_N];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int k_tile_idx = 0; k_tile_idx < k; k_tile_idx += WMMA_K) {
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int float4_idx = tid + i * warpSize;
            int row = float4_idx / 4;
            int col = (float4_idx % 4) * 4;

            if (out_tile_row + row < h && k_tile_idx + col < k) {
                float4 val = *(reinterpret_cast<const float4*>(&a[(out_tile_row + row) * k + k_tile_idx + col]));
                ((half2*)(&tile_a[row][col]))[0] = __float22half2_rn(make_float2(val.x, val.y));
                ((half2*)(&tile_a[row][col]))[1] = __float22half2_rn(make_float2(val.z, val.w));
            } else {
                ((half2*)(&tile_a[row][col]))[0] = make_half2(__float2half_rn(0.0f), __float2half_rn(0.0f));
                ((half2*)(&tile_a[row][col]))[1] = make_half2(__float2half_rn(0.0f), __float2half_rn(0.0f));
            }
            
            if (k_tile_idx + row < k && out_tile_col + col < w) {
                float4 val = *(reinterpret_cast<const float4*>(&b[(k_tile_idx + row) * w + out_tile_col + col]));
                ((half2*)(&tile_b[row][col]))[0] = __float22half2_rn(make_float2(val.x, val.y));
                ((half2*)(&tile_b[row][col]))[1] = __float22half2_rn(make_float2(val.z, val.w));
            } else {
                ((half2*)(&tile_b[row][col]))[0] = make_half2(__float2half_rn(0.0f), __float2half_rn(0.0f));
                ((half2*)(&tile_b[row][col]))[1] = make_half2(__float2half_rn(0.0f), __float2half_rn(0.0f));
            }
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &tile_a[0][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &tile_b[0][0], WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    if (out_tile_row < h && out_tile_col < w) {
        wmma::store_matrix_sync(c + out_tile_row * w + out_tile_col, c_frag, w, wmma::mem_row_major);
    }
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
