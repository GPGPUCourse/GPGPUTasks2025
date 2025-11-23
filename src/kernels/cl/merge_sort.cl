#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const Prim * src,
    __global       Prim * dst,
             const uint   iter,
             const uint   size
) {
    const uint index = get_global_id(0);

    if (index >= size) {
        return;
    }

    const uint len = 1u << (iter - 1);
    const uint left_bound = (index >> iter) << iter;
    const uint middle = left_bound + len;
    const Prim prim = src[index];

    if (middle >= size) {
        dst[index] = prim;
        return;
    }

    uint target = prim.morton;
    uint left, right, const_left, offset;
    if (index < middle) {
        offset = left_bound;
        const_left = left = middle - 1;
        right = min(middle + len, size);
    } else {
        offset = middle;
        const_left = left = left_bound - 1;
        right = middle;
        target += 1;
    }

    while (right - left > 1) {
        const uint mid = (left + right) / 2;
        const uint cmp = src[mid].morton < target;
        const uint ncmp = 1 - cmp;
        left = cmp * mid + ncmp * left;
        right = ncmp * mid + cmp * right;
    }

    const uint dst_index = left_bound + (index - offset) + (left - const_left);
    dst[dst_index] = prim;
}
