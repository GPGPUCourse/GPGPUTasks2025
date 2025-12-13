#include "../defines.h"
#include "helpers/rassert.cl"

#include "../shared_structs/morton_code_gpu_shared.h"

__kernel void get_sorted_morton_codes(
    __global const uint *triIndexes,
    __global const MortonCode *mortonCodes,
    const uint nfaces,
    __global MortonCode *sortedCodes)
{
    const uint i = get_global_id(0);
    if (i >= nfaces) {
        return;
    }
    sortedCodes[i] = mortonCodes[triIndexes[i]];
}