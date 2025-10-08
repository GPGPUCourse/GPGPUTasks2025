#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

#define WX 16
#define WY 16

__attribute__((reqd_work_group_size(WX, WY, 1)))
__kernel void
matrix_02_transpose_coalesced_via_local_memory(
    __global const float* matrix, // w x h
    __global float* transposed_matrix, // h x w
    unsigned int w,
    unsigned int h)
{
    // const uint mask = WX * WY - 1;

    const uint xmask = WX - 1;
    const uint ymask = WY - 1;
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    uint i = get_local_id(0);
    uint j = get_local_id(1);

    uint gx = get_group_id(0);
    uint gy = get_group_id(1);

    __local float wg_data[WX * WY];

    // version without bank conflict resolution, result: 8.2 GB/s

    // wg_data [WX * j + i] = matrix[y * w + x];
    // barrier(CLK_LOCAL_MEM_FENCE);
    // transposed_matrix [(gx * 16 + j) * h + (gy * 16 + i)] = wg_data[WY * i + j];

    // version with resolved bank conflicts, result: 8.38 GB/s )))))))))

    // if (gx == 0 && gy == 0 && i <= 1 && j <= 1) {
    //     printf ("%u %u:load to %u; save from %u\n", i,j,(WX * (j) + ((i + j) & xmask)) ,(WX * i + ((i + j) & ymask)));
    // }
    if (x >= w || y >= h)
        wg_data[(WX * (j) + (i + j & xmask))] = 0.f;
    else
        wg_data[(WX * (j) + (i + j & xmask))] = matrix[y * w + x];
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((gx * 16 + j) >= w || (gy * 16 + i) >= h)
        return;
    transposed_matrix[(gx * 16 + j) * h + (gy * 16 + i)] = wg_data[(WX * i + (i + j & ymask))];
}
