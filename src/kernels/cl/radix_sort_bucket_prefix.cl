#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_bucket_prefix(
    __global const uint* bucket_size,
    __global uint* bucket_base)
{
    if (get_global_id(0) == 0) {
        uint sum = 0;
        for (uint i = 0; i < RAD_SIZE; ++i) {
            bucket_base[i] = sum;
            sum += bucket_size[i];
        }
    }
}
