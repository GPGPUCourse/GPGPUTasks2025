#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void small_merge_sort(
    __global const Prim * global_src,
    __global       Prim * global_dst,
             const uint   size
) {
    const uint global_index = get_global_id(0);

    if (global_index >= size) {
        return;
    }

    __local Prim buf1[GROUP_SIZE];
    __local Prim buf2[GROUP_SIZE];
    __local Prim* src = buf1;
    __local Prim* dst = buf2;

    const uint index = get_local_id(0);
    src[index] = global_src[global_index];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint iter = 1; iter < PIVOT; ++iter) {
        const uint len = 1u << (iter - 1);
        const uint left_bound = (index >> iter) << iter;
        const uint middle = left_bound + len;

        uint target = src[index].morton;
        uint left, right, const_left, offset;
        if (index < middle) {
            offset = left_bound;
            const_left = left = middle - 1;
            right = min(middle + len, (uint)GROUP_SIZE);
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
        dst[dst_index] = src[index];
        barrier(CLK_LOCAL_MEM_FENCE);

        {
            __local unsigned int* tmp = dst;
            dst = src;
            src = tmp;
        }
    }

    global_dst[global_index] = src[index];
}
