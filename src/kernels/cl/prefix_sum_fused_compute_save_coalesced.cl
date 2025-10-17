#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

uint calcNextSum(__global const uint* pow2_sum,
                 const uint curN,
                 const uint x) {
    if (x * 2 < curN) {
        // printf("curN: %d   sumIdx: %d   fst: %d   snd: %d\n",
        //     curN, x, pow2_sum[x * 2], (x * 2 + 1 < curN ? pow2_sum[x * 2 + 1] : 0));
        return pow2_sum[x * 2] + (x * 2 + 1 < curN ? pow2_sum[x * 2 + 1] : 0);
    }
    return 0;
}

void writeNextPow(__global uint* next_pow2_sum,
                const uint* sum,
                const uint sumIdx,
                const uint curN,
                const uint x,
                const uint newX,
                const uint cntBlocks,
                const uint offset) {
    const uint idx = ((x < offset) ? x : newX / cntBlocks);
    if (sumIdx + idx < (curN + 1) / 2 && idx < 2) {
        // printf("x: %d\tidx: %d\tsumIdx: %d\tsum: %d\n",
        //     x, idx, sumIdx, sum[idx]);
        next_pow2_sum[sumIdx + idx] = sum[idx];
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_fused_compute_save_coalesced(
    __global const uint* pow2_sum, // contains curN values
    __global       uint* next_pow2_sum, // will contain (curN+1)/2 values
    __global       uint* prefix_sum_accum,
    unsigned int curN,
    unsigned int n, // work range must be at least (n + 1) / 2
    unsigned int pow2) // at least 1
{
    const uint x = get_global_id(0);
    const uint range = (1 << pow2);
    uint cntBlocks = (n + 1) / (range << 1);
    uint offset = (n + 1) % (range << 1);
    if (offset > range) {
        offset -= range;
        // ++cntBlocks;
        // if (cntBlocks * range > (n + 1) / 2) {
        //     printf("wtf\n");
        // }
    } else {
        offset = 0;
    }

    uint idx;
    uint sumIdx;
    uint newX = x - offset;
    if (x < offset) {
        idx = n - x - 1;
        sumIdx = cntBlocks * 2;
        // printf("pow2: %d   offset: %d   x: %d   idx: %d   sumIdx: %d\n",
        //     pow2, offset, x, idx, sumIdx);
    } else {
        if (newX / cntBlocks >= range) {
            return;
        }
        const uint blockNum = newX % cntBlocks;
        sumIdx = blockNum * 2;
        const uint begIdx = sumIdx * range + range - 1;  
        idx = begIdx + (newX / cntBlocks);
    }


    uint sum[2] = {calcNextSum(pow2_sum, curN, sumIdx),
        calcNextSum(pow2_sum, curN, sumIdx + 1)};

    if (idx < n) {
        prefix_sum_accum[idx] += sum[0];
        // printf("pow2: %d   idx: %d   sum: %d   sumIdx: %d   res: %d\n", 
        //     pow2, idx, sum[0], sumIdx, prefix_sum_accum[idx]);
    }

    if (x == 0 && curN % 2 == 1) {
        next_pow2_sum[curN / 2] = pow2_sum[curN - 1];
    }



    writeNextPow(next_pow2_sum, sum, sumIdx, curN, x, newX, cntBlocks, offset);
    // writeNextPow(next_pow2_sum, sum, sumIdx + 1, curN, x, newX, cntBlocks);
}
