#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define WG_W (16)
#define WG_H (16)

__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const int global_j = get_global_id(1);
    const int global_i = get_global_id(0);
    const int local_j = get_local_id(1); 
    const int local_i = get_local_id(0);

    __local float a_fragment[WG_W][WG_H];
    __local float b_fragment[WG_W][WG_H];

    if (global_i >= w || global_j >= h) {
        return;
    }

    float sum_el = 0;

    const uint num_fragments = k / WG_W + ((k % WG_W == 0) ? 0 : 1);
    for (uint fragment_id = 0; fragment_id < num_fragments; ++fragment_id) {
        const uint a_fragment_i = fragment_id * WG_W + local_i;
        const uint a_fragment_j = global_j;
        if (a_fragment_i < k && a_fragment_j < h) {
            a_fragment[local_j][local_i] = a[a_fragment_j * k + a_fragment_i];
        } else {
            a_fragment[local_j][local_i] = 0;
        }

        const uint b_fragment_i = global_i;
        const uint b_fragment_j = fragment_id * WG_W + local_j;
        if (b_fragment_i < w && b_fragment_j < k) {
            b_fragment[local_j][local_i] = b[b_fragment_j * w + b_fragment_i];
        } else {
            b_fragment[local_j][local_i] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint t = 0; t < WG_W; ++t) {
            sum_el += a_fragment[local_j][t] * b_fragment[t][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

    }

    c[global_j * w + global_i] = sum_el;
}
