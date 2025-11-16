#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_01_local_counting(
    __global const uint* input,
    __global uint* group_counts,
    unsigned int offset,
    unsigned int n)
{
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint idx = lid + gid * GROUP_SIZE;
    const uint n_of_buckets = 1u << BITS_PER_PASS;

    __local uint buffer[GROUP_SIZE];
    buffer[lid] = idx < n ? input[idx] : 0;
    if (lid < n_of_buckets) {
        group_counts[n_of_buckets * gid + lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        const uint remaining = min((int)( n - gid * GROUP_SIZE), GROUP_SIZE);
        for (uint i = 0; i < remaining; ++i) {
            const uint mask = (n_of_buckets - 1);
            const uint num = (buffer[i] >> offset) & mask;
            group_counts[n_of_buckets * gid + num]++;
        }
    }
}
