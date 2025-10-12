#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_04_multiply_via_local_memory(
    __global const float* a, // rows=h x cols=k
    __global const float* b, // rows=k x cols=w
    __global float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{

    const unsigned int index_w = get_global_id(0);
    const unsigned int index_h = get_global_id(1);

    const unsigned int index_w_local = get_local_id(0);
    const unsigned int index_h_local = get_local_id(1);

    __local float local_data_a[LOCAL_MEM_SIZE_X][LOCAL_MEM_SIZE_Y + 1];
    __local float local_data_b[LOCAL_MEM_SIZE_X][LOCAL_MEM_SIZE_Y + 1];

    float summ = 0.0f;

    for (unsigned int i = 0; i < k; i += GROUP_SIZE_X) {
        // закидываем A в локальную память
        if ((index_w_local + i) >= k || index_h >= h) {
            local_data_a[index_h_local][index_w_local] = 0;
        } else {
            // тут, конечно, не совсем coalesced доступ к памяти, а в 128 / GROUP_SIZE_X раз хуже
            local_data_a[index_h_local][index_w_local] = a[(index_w_local + i) + k * index_h];
        }

        // закидываем B в локальную память
        if ((index_h_local + i) >= k || index_w >= w) {
            local_data_b[index_h_local][index_w_local] = 0;
        } else {
            local_data_b[index_h_local][index_w_local] = b[index_w + w * (index_h_local + i)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (index_w < w && index_h < h) {
            for (unsigned int j = 0; (j < GROUP_SIZE_X); j++) {
                // кажется, без банк конфликтов не обошлось...
                summ += local_data_a[index_h_local][j] * local_data_b[j][index_w_local];
            }
        }
    }
    if (index_w < w && index_h < h) {
        c[index_w + w * index_h] = summ;
    }
}
