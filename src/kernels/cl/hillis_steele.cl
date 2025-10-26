#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void hillis_steele(
    __global const uint* input,
    __global uint* output,
    unsigned int n,
    unsigned int stride)
{
    const unsigned int i = get_global_id(0);
    
    if (i < n) {
        if (i >= stride) {
            output[i] = input[i] + input[i - stride];
        } else {
            output[i] = input[i];
        }
    }
}