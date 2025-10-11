#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

// Было подсмотренно где-то в интернете, скорее всего подсказал AI Overview в поске Google
inline void atomicAdd_f(volatile __global float *addr, float val) {
    float old_val;
    do {
        old_val = *addr;
    } while (atomic_cmpxchg((volatile __global unsigned int *)addr, as_uint(old_val), as_uint(old_val + val)) != as_uint(old_val));
}

__attribute__((reqd_work_group_size(PACK_SIZE, PACK_SIZE, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const    float* a, // rows=h x cols=k
                       __global const    float* b, // rows=k x cols=w
                       __global volatile float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float local_data_a[PACK_SIZE * PACK_SIZE];
    __local float local_data_b[PACK_SIZE * PACK_SIZE];

    const unsigned int global_x = get_global_id(0);
    const unsigned int global_y = get_global_id(1);

    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);

    const unsigned int a_x = global_x;
    const unsigned int a_y = global_y;

    if (a_x < k && a_y < h) {
        local_data_a[local_x + PACK_SIZE * local_y] = a[a_x + k * a_y];
    } else {
        local_data_a[local_x + PACK_SIZE * local_y] = 0;
    }

    for (unsigned int j = 0; j < w; j += PACK_SIZE) {
        const unsigned int b_x = j + local_x;
        const unsigned int b_y = global_x - local_x + local_y;

        if (b_x < w && b_y < k) {
            local_data_b[local_x + PACK_SIZE * local_y] = b[b_x + w * b_y];
        } else {
            local_data_b[local_x + PACK_SIZE * local_y] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        float acc = 0;
        #pragma unroll
        for (unsigned int i = 0; i < PACK_SIZE; ++i) {
            acc += local_data_a[i + PACK_SIZE * local_y] * local_data_b[i * PACK_SIZE + local_x];
        }

        const unsigned int c_x = b_x;
        const unsigned int c_y = a_y;

        if (c_x < w && c_y < h) {
            atomicAdd_f(&c[c_x + w * c_y], acc);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
