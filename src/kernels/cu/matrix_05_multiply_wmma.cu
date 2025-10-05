#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
#include <mma.h>
using namespace nvcuda;

const int WMMA_SZ = 16;

// preconditions: w,h,k % 16 == 0
__global__ void matrix_multiply_wmma(
                       const __half* a, // rows=h x cols=k
                       const __half* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    unsigned int warpX = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    unsigned int warpY = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, WMMA_SZ, WMMA_SZ, WMMA_SZ, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_SZ, WMMA_SZ, WMMA_SZ, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_SZ, WMMA_SZ, WMMA_SZ, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int i = 0; i < k; i += WMMA_SZ) {
        unsigned int aRow = warpY * WMMA_SZ;    // the topmost row
        unsigned int aCol = i;                  // the leftmost column

        unsigned int bRow = i;                  // the topmost row
        unsigned int bCol = warpX * WMMA_SZ;    // the leftmost column

        if (aRow < h && aCol < k && bCol < w && bRow < k) {
            wmma::load_matrix_sync(a_frag, a + aRow * k + aCol, k);
            wmma::load_matrix_sync(b_frag, b + bRow * w + bCol, w);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    unsigned int cRow = warpY * WMMA_SZ;    // the topmost row
    unsigned int cCol = warpX * WMMA_SZ;    // the leftmost column

    if (cRow < h && cCol < w) {
        wmma::store_matrix_sync(c + cRow * w + cCol, acc_frag, w, wmma::mem_row_major);
    }
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_16f &a, const gpu::gpu_mem_16f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda

