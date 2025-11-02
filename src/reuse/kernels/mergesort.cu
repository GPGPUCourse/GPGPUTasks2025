#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "../wrappers.h"
#include "helpers/rassert.cu"

__device__ int search(const unsigned int* l, const unsigned int* r, int size, int d) {
    
    int L = 0, R = d + 1;
    if (d >= size) {
        L = d - size;
        R = size + 1;
    }
    while (R - L > 1) {
        int m = (R + L) / 2;
        int a = l[m - 1];
        int b = r[d - m ];
        if (a <= b) {
            L = m;
        } else {
            R = m;
        }
    }
    return L;
}

__global__ void mergesort(const unsigned int* a, unsigned int* c, unsigned int size, unsigned int n) {
    int x = threadIdx.x;
    int i = blockIdx.x * blockDim.x + x;
    int g = i / (2 * size);
    int d = i % (2 * size);
    const unsigned int* l = a + g * 2 * size;
    const unsigned int* r = l + size;

    int ind = search(l, r, size, d);
    int li = ind;
    int ri = d - ind;

    if (li == size) {
        c[i] = r[ri];
    } else if (ri == size) {
        c[i] = l[li];
    } else {
        if (l[li] <= r[ri]) {
            c[i] = l[li];
        } else {
            c[i] = r[ri];
        }
    }
}

namespace cuda {
void mergesort(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 c, unsigned int size, unsigned int n) {
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::mergesort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), c.cuptr(), size, n);
    CUDA_CHECK_KERNEL(stream);
}
}  // namespace cuda
