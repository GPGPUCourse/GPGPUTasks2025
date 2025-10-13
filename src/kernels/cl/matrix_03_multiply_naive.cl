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
__kernel void matrix_03_multiply_naive(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    Index p = {get_global_id(0), get_global_id(1), w, h};

    if (id(p) >= w * h)
        return;

    float sum = 0;
    for (unsigned int ik = 0; ik < k; ik++) {
        Index a_id = {ik, p.y, k, h};
        Index b_id = {p.x, ik, w, k};
        sum += a[id(a_id)] * b[id(b_id)];
    }

    c[id(p)] = sum;
}
