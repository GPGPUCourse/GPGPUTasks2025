#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

struct Coords {
    unsigned int i;
    unsigned int j;
};

struct Coords get_image(const unsigned int ord) {
    const unsigned int img_i = ord % GROUP_SIZE_Y;
    const unsigned int img_j = (img_i + (ord - img_i) % GROUP_SIZE_X);

    struct Coords result;
    result.i = img_i;
    result.j = img_j;
    return result;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1);

    if (i >= h || j >= w) {
        return;
    }

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    const unsigned int ord = local_i * GROUP_SIZE_X + local_j;
    const struct Coords source_image = get_image(ord);

    __local float tmp[GROUP_SIZE_Y][GROUP_SIZE_X];

    tmp[source_image.i][source_image.j] = matrix[i * w + j];

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int transposed_ord = local_j * GROUP_SIZE_Y + local_i;
    const struct Coords target_image = get_image(transposed_ord);

    transposed_matrix[j * h + i] = tmp[target_image.i][target_image.j];
}
