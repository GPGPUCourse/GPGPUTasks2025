#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_tiles(
    __global uint* input_output_data,
    __global uint* sparse_buf
) {
    uint lbound = get_group_id(0) * TILE_SIZE;
    uint rbound = lbound + TILE_SIZE;

    uint lid = get_local_id(0);
    __local uint local_data[TILE_SIZE];
    __local uint temp_data[TILE_SIZE];
    for (int shift = 0; shift < TILE_SIZE; shift += GROUP_SIZE) {
        local_data[shift + lid] = input_output_data[lbound + shift + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint width = 2; width <= TILE_SIZE; width *= 2) {
        for (uint shift = 0; shift < TILE_SIZE; shift += 2 * GROUP_SIZE) {
            uint l = (shift + 2 * lid) / width * width;
            uint r = l + width;
            uint m = (l + r) / 2;
            uint i = (shift + 2 * lid - l) / 2;

            uint lhs = -1, rhs = width / 2;
            uint elem = local_data[l + i];
            while (rhs - lhs > 1) {
                uint mid = (lhs + rhs) / 2;
                if (local_data[m + mid] >= elem) {
                    rhs = mid;
                } else {
                    lhs = mid;
                }
            }
            temp_data[l + i + rhs] = elem;
            
            lhs = -1; rhs = width / 2;
            elem = local_data[m + i];
            while (rhs - lhs > 1) {
                uint mid = (lhs + rhs) / 2;
                if (local_data[l + mid] > elem) {
                    rhs = mid;
                } else {
                    lhs = mid;
                }
            }
            temp_data[l + i + rhs] = elem;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int shift = 0; shift < TILE_SIZE; shift += GROUP_SIZE) {
            local_data[shift + lid] = temp_data[shift + lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int shift = 0; shift < TILE_SIZE; shift += GROUP_SIZE) {
        input_output_data[lbound + shift + lid] = local_data[shift + lid];
        if ((shift + lid) % WRITE_EVERY == 0) {
            sparse_buf[(lbound + shift + lid) / WRITE_EVERY] = local_data[shift + lid];
        }
    }
}
