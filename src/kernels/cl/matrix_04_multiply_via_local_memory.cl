#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

typedef struct{
    unsigned int x, y, w, h;
} Index;

// returns abs. Index in row wise format.
inline unsigned int id(const Index p) { return p.y * p.w + p.x; }
inline Index transpose(const Index p) {
    Index q = {p.y, p.x, p.w, p.h};
    return q;
}

// __attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    Index global_p = {get_global_id(0), get_global_id(1), w, h};
    Index local_p = {get_local_id(0), get_local_id(1), GROUP_SIZE_X, GROUP_SIZE_Y};
    __local float matrix_a_cache[GROUP_SIZE];
    __local float matrix_b_cache[GROUP_SIZE];

    c[id(global_p)] = 0;

    for (unsigned int offset = 0; offset < k; offset += GROUP_SIZE_X) {
        Index a_id = {local_p.x + offset, global_p.y, k, h};
        Index b_id = {global_p.x        , local_p.y + offset, w, k};

        // printf("global       : %u %u --> %u\n", global_p.x, global_p.y, id(global_p));
        // printf("local        : %u %u --> %u\n", local_p.x, local_p.y, id(local_p));
        // printf("block id     : %u\n", block);
        // printf("a_id         : %u %u --> %u %f\n", a_id.x, a_id.y, id(a_id), a[id(a_id)]);
        // printf("b_id         : %u %u --> %u %f\n", b_id.x, b_id.y, id(b_id), b[id(b_id)]);

        matrix_a_cache[id(local_p)] = a[id(a_id)];
        matrix_b_cache[id(local_p)] = b[id(b_id)];

        barrier(CLK_LOCAL_MEM_FENCE);

        // if (local_p.x == 0 && local_p.y == 1)
        //     printf("Cachce A\n%f %f\n%f %f\n",matrix_a_cache[0], matrix_a_cache[1], matrix_a_cache[2], matrix_a_cache[3]);

        for (unsigned int i = 0; i < GROUP_SIZE_X; ++i) {
            Index a_cache_id = {i, local_p.y, GROUP_SIZE_X, GROUP_SIZE_Y};
            Index b_cache_id = {local_p.x, i, GROUP_SIZE_X, GROUP_SIZE_Y};
            // if (id(global_p) == 1)
                // printf("block : %u, local : %u %u, for pos %u dot %f %f\n", block, local_p.x, local_p.y, id(global_p),  matrix_a_cache[id(a_cache_id)], matrix_b_cache[id(b_cache_id)]);
                // printf("Cachce A\n%f %f\n%f %f\n",matrix_a_cache[0], matrix_a_cache[1], matrix_a_cache[2], matrix_a_cache[3]);
            c[id(global_p)] += matrix_a_cache[id(a_cache_id)] * matrix_b_cache[id(b_cache_id)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
