#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

inline uint upper_bound_le(
    __global const uint* arr,
                   uint left,
                   uint right,
                   uint a
    ) {
    uint l = left, r = right;
    uint mid, val;
    while (l < r) {
        mid = l + ((r - l) >> 1);
        val = arr[mid];
        if (val <= a) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

inline uint upper_bound_lt(
    __global const uint* arr,
    uint left,
    uint right,
    uint a
) {
    uint l = left, r = right;
    uint mid, val;
    while (l < r) {
        mid = l + ((r - l) >> 1);
        val = arr[mid];
        if (val < a) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* last_layer,
    __global       uint* new_layer,
             const uint n,
                   uint pow) {

    const uint id = get_global_id(0);
    if (id < n) {
        const uint block_id = id >> pow;
        const uint a = last_layer[id];
        uint l_inclusive, r_exclusive, offset;
        if (block_id & 1u) { // odd <- '<='
            l_inclusive = (1u << pow) * (block_id - 1u);
            r_exclusive = l_inclusive + (1u << pow);
            // TODO find how many in [l_inclusive;r_exclusive) less or equal than last_layer[id]

            uint pos = upper_bound_le(last_layer, l_inclusive, r_exclusive, a);
            uint count_le = pos - l_inclusive;
            new_layer[l_inclusive + id - r_exclusive + count_le] = a;
        } else { // even -> '<'
            l_inclusive = (1u << pow) * (block_id + 1u);
            if (l_inclusive + 1 >= n) { // handle bad n
                new_layer[id] = last_layer[id];
                return;
            }
            r_exclusive = l_inclusive + (1u << pow);
            if (r_exclusive > n) { // whether out of bounds
                r_exclusive = n;
            }
            // TODO find how many in [l_inclusive;r_exclusive) greater than last_layer[id]

            uint pos = upper_bound_lt(last_layer, l_inclusive, r_exclusive, a);
            uint count_lt = pos - l_inclusive;
            new_layer[id + count_lt] = a;

        }
    }
}
