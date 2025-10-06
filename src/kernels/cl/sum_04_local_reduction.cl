#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define WARP_SIZE 32

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_04_local_reduction(__global const uint* a,
                                     __global       uint* b,
                                            unsigned int  n)
{
    // Подсказки:
    // const uint index = get_global_id(0);
    // const uint local_index = get_local_id(0);
    // __local uint local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);

    // TODO

    const uint index = get_global_id(0);
    
    const uint local_index = get_local_id(0);
    __local uint local_data[GROUP_SIZE];
    const uint warpsCnt = GROUP_SIZE / WARP_SIZE;
    __local uint localReduction[warpsCnt];

    local_data[local_index] = (index < n ? a[index] : 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index % WARP_SIZE == 0) {
        uint localSum = 0;
        for (uint i = 0; i < WARP_SIZE; ++i) {
            localSum += local_data[local_index + i];
        }
        // printf("warp idx: %d   sum: %d\n", index, localSum);
        localReduction[local_index / WARP_SIZE] = localSum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index == 0) {
        uint localSum = 0;
        for (uint i = 0; i < warpsCnt; ++i) {
            localSum += localReduction[i];
        }
        // printf("group idx: %d   sum: %d\n", index, localSum);
        b[index / GROUP_SIZE] = localSum;    
    }
}
