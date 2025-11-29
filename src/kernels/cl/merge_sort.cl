#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

inline uint upper_bound_cmp(
    __global const uint* arr,
                   uint l,
                   uint r,
                   uint val,
                   bool in) {

    uint mid, v;
    while (l < r) {
        mid = l + ((r - l) >> 1);
        v = arr[mid];
        if (in ? (v <= val) : (v < val))
            l = mid + 1;
        else
            r = mid;
    }
    return l;

}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* last_layer,
    __global       uint* new_layer,
             const uint n,
                   uint pow) {

    uint id = get_global_id(0);
    if (id >= n) return;

    uint val = last_layer[id];
    uint block_size = 1u << pow;
    uint block_id   = id >> pow;
    uint pos_in_block = id & (block_size - 1u);

    uint l_start, r_start, l_end, r_end;
    uint cnt, pos;

    if (block_id & 1u) {
        l_start = block_size * (block_id - 1u);
        l_end   = l_start + block_size;

        pos = upper_bound_cmp(last_layer, l_start, l_end, val, true);
        cnt = pos - l_start;
    } else {
        l_start = block_size * block_id;
        r_start = block_size + l_start;
        if (r_start >= n) {
            new_layer[id] = val;
            return;
        }
        r_end = r_start + block_size;
        if (r_end > n) r_end = n;

        pos = upper_bound_cmp(last_layer, r_start, r_end, val, false);
        cnt = pos - r_start;
    }
    new_layer[l_start + cnt + pos_in_block] = val;
}
