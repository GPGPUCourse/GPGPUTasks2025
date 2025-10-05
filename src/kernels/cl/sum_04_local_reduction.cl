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
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint group_index = get_group_id(0);

    __local uint local_data[GROUP_SIZE];

    if(index < n ){
        local_data[local_index] = a[index];
    } else {
        local_data[local_index] =0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = GROUP_SIZE/2; offset > 0; offset = offset/2) {
	      
        if (local_index < offset) {

            local_data[local_index] += local_data[local_index + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_index==0){
    b[group_index]=local_data[0];
    }
    // TODO
}
