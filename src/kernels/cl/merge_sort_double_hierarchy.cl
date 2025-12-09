#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_double_hierarchy(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k, // must be more than GROUP_SIZE
                   int  n)
{
    const unsigned int i = get_global_id(0);
    const int sortedK2 = sorted_k * 2;
    const int block = i / sortedK2;
    const int iInBlock = i % sortedK2;

    const int blockStart = block * sorted_k;

    int xStart = sorted_k * block;
    int yStart = xStart;

    if (iInBlock >= sorted_k) {
        xStart += sorted_k - 1;
        yStart += iInBlock - sorted_k + 1;
    } else {
        xStart += iInBlock;
    }

    const uint localI = get_local_id(0); 

    __local int xBorder, yBorder;
    const int mx = sorted_k - abs(sorted_k - (int)(iInBlock + 1));
    xBorder = xStart;
    yBorder = yStart + mx - 1;

    barrier(CLK_LOCAL_MEM_FENCE);

    int x, y;
    if (localI == GROUP_SIZE - 1 && i < n) {
        int l = -1, r = mx;
        while (r - l > 1) {
            const int md = (l + r) / 2;

            int xTmp = xStart - md, yTmp = yStart + md;
            const int realX = (xTmp / sorted_k) * sorted_k * 2 + xTmp % sorted_k;
            const int realY = sorted_k + (yTmp / sorted_k) * sorted_k * 2 + yTmp % sorted_k;
            if (realX < n && realY < n && input_data[realX] > input_data[realY]) {
                l = md;
            } else {
                r = md;
            }
        }

        x = xStart - l - 1;
        y = yStart + l;
        if (y >= blockStart && 
            (x < blockStart || 
                (input_data[(x / sorted_k) * sorted_k * 2 + x % sorted_k] <= input_data[sorted_k + (y / sorted_k) * sorted_k * 2 + y % sorted_k]))) {
            ++x;
        } else {
            ++y;
        }
        xBorder = x;
        yBorder = y;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localI != GROUP_SIZE - 1 && i < n) {
        int l = max(xStart - xBorder - 1, -1);
        int r = min(yBorder - yStart + 1, mx);

        while (r - l > 1) {
            const int md = (l + r) / 2;

            int xTmp = xStart - md, yTmp = yStart + md;
            const int realX = (xTmp / sorted_k) * sorted_k * 2 + xTmp % sorted_k;
            const int realY = sorted_k + (yTmp / sorted_k) * sorted_k * 2 + yTmp % sorted_k;
            if (realX < n && realY < n && input_data[realX] > input_data[realY]) {
                l = md;
            } else {
                r = md;
            }
        }

        x = xStart - l - 1;
        y = yStart + l;
        if (y >= blockStart && 
            (x < blockStart || 
                (input_data[(x / sorted_k) * sorted_k * 2 + x % sorted_k] <= input_data[sorted_k + (y / sorted_k) * sorted_k * 2 + y % sorted_k]))) {
            ++x;
        } else {
            ++y;
        }
    }

    if (i < n) {
        const int realX = (x / sorted_k) * sorted_k * 2 + x % sorted_k;
        const int realY = sorted_k + (y / sorted_k) * sorted_k * 2 + y % sorted_k;
        
        const uint inX = ((x - blockStart < sorted_k && realX < n && x >= 0) ? input_data[realX] : UINT_MAX);
        const uint inY = ((y - blockStart < sorted_k && realY < n) ? input_data[realY] : UINT_MAX);
        uint out;
        uint inIdx;
        if (inX <= inY) {
            inIdx = x - blockStart;
            out = inX;
        } else {
            inIdx = y - blockStart;
            out = inY;
        }
        
        if (inIdx < sorted_k){
            output_data[x + y] = out;
        }
    }
}
