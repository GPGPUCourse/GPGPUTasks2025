#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global uint* pow2_sum,
    unsigned int levels,
    unsigned int n)
{
    __local uint buffer[2 * GROUP_SIZE - 1];
    unsigned int offset = n - 1;

    unsigned int index = get_global_id(0);
    unsigned int local_index = get_local_id(0);
    unsigned int lci_offset = GROUP_SIZE - 1;
    unsigned int take_from_group = GROUP_SIZE;

    if (index < n) {
        buffer[lci_offset + local_index] = pow2_sum[offset + index];
    } else {
        buffer[lci_offset + local_index] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int i = 1; i <= levels; i++) {
        offset = (offset - 1) >> 1;
        lci_offset = (lci_offset - 1) >> 1;
        n >>= 1;
        take_from_group >>= 1;

        unsigned int pow2_index = local_index + get_group_id(0) * take_from_group;

        if (local_index < take_from_group && pow2_index < n) {
            unsigned int lci = lci_offset + local_index;
            unsigned int tmp = buffer[(lci << 1) + 1] + buffer[(lci << 1) + 2];


            buffer[lci] = tmp;
            pow2_sum[offset + pow2_index] = tmp;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
