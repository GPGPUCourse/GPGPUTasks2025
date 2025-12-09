#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((always_inline))
void pairByDiagAndNum(const int xStart, const int yStart, const int i, 
    __private int* resX, 
    __private int* resY) {
    *resX = xStart - i;
    *resY = yStart + i;
}

__attribute__((always_inline))
int toRealIdxX(const int x, const int sorted_k, const int block) {
    return (x / sorted_k) * sorted_k * 2 + x % sorted_k;
}

__attribute__((always_inline))
int toRealIdxY(const int y, const int sorted_k, const int block) {
    return sorted_k + (y / sorted_k) * sorted_k * 2 + y % sorted_k;
}


__attribute__((always_inline))
void binarySearch(
    int l, int r, int xStart, int yStart, int sorted_k, int block, int n, int blockStart,
    __private int* x, __private int* y, __global const uint* input_data) {
    while (r - l > 1) {
        const int md = (l + r) / 2;

        int xTmp, yTmp;
        pairByDiagAndNum(xStart, yStart, md, &xTmp, &yTmp);
        const int realX = toRealIdxX(xTmp, sorted_k, block);
        const int realY = toRealIdxY(yTmp, sorted_k, block);
        if (realX < n && realY < n && input_data[realX] > input_data[realY]) {
            l = md;
        } else {
            r = md;
        }
    }

    pairByDiagAndNum(xStart, yStart, l, x, y);
    *x -= 1;
    if (*y >= blockStart && 
        (*x < blockStart || (input_data[toRealIdxX(*x, sorted_k, block)] <= input_data[toRealIdxY(*y, sorted_k, block)]))) {
        *x += 1;
    } else {
        *y += 1;
    }
}

__attribute__((always_inline))
void writeToOutput(int x, int y, int sorted_k, int block, int blockStart, int n,
    __global const uint* input_data, __global uint* output_data) {
    const int realX = toRealIdxX(x, sorted_k, block);
    const int realY = toRealIdxY(y, sorted_k, block);
    // if (x - blockStart < sorted_k && realX < n) {
    //     rassert(realX >= 0, 7673645);
    //     if (realX < 0) {
    //         printf("%d %d\n", x, y);
    //     }
    // }
    
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

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_double_hierarchy(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k, // must be more than GROUP_SIZE
                   int  n)
{
    const unsigned int i = get_global_id(0);
    // if (i >= n) {
    //     return;
    // }
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

    // rassert((i - localI) / sortedK2 == block, 8989);

    __local int xBorder, yBorder;
    const int mx = sorted_k - abs(sorted_k - (int)(iInBlock + 1));
    xBorder = xStart;
    yBorder = yStart + mx - 1;

    barrier(CLK_LOCAL_MEM_FENCE);

    int x, y;
    if (localI == GROUP_SIZE - 1 && i < n) {
        binarySearch(-1, mx, 
            xStart, yStart, sorted_k, block, n, blockStart,
            &x, &y, input_data);
        xBorder = x;
        yBorder = y;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localI != GROUP_SIZE - 1 && i < n) {
        int l = max(xStart - xBorder - 1, -1);
        int r = min(yBorder - yStart + 1, mx);

        // printf("%d %d   %d %d\n", xStart, xBorder, yStart, yBorder);
        // printf("new borders: -1 | %d %d | %d\n\n", l, r, mx);
        binarySearch(l, r, 
            xStart, yStart, sorted_k, block, n, blockStart,
            &x, &y, input_data);

        int rightX, rightY;
        binarySearch(-1, mx,
            xStart, yStart, sorted_k, block, n, blockStart,
            &rightX, &rightY, input_data);
        // if (x != rightX || y != rightY) {
        //     printf("%d %d   %d %d\nwrong: %d %d   right: %d %d\nnew borders: -1 | %d %d | %d\n\n", 
        //         xStart, xBorder, yStart, yBorder,
        //         x, y, rightX, rightY,
        //         l, r, mx);
        // }
    }

    if (i < n) {
        writeToOutput(x, y, sorted_k, block, blockStart, n, input_data, output_data);
    }
}
