#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void copy(
    __global uint* src,
    __global uint* dst,
    unsigned int n)
{
    size_t i = get_global_id(0);
    if (i >= n) return;
    dst[i] = src[i];
}
