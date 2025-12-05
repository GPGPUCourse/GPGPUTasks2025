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

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    // printf("i: %5d\t2n: %5d\n", i, n * 2);
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

    const int mx = sorted_k - abs(sorted_k - (int)(iInBlock + 1));
    // printf("(iInBlock + 1) = %d\t abs(sorted_k - (iInBlock + 1)) = %d\n", 
    //     (iInBlock + 1), abs(sorted_k - (iInBlock + 1)));
    int l = -1, r = mx; // в l первый > второго
    while (r - l > 1) {
        const int md = (l + r) / 2;

        int x, y;
        pairByDiagAndNum(xStart, yStart, md, &x, &y);
        const int realX = toRealIdxX(x, sorted_k, block);
        const int realY = toRealIdxY(y, sorted_k, block);
        // if (sorted_k == 2 && i == 1) {
            // printf("sorted_k %d i %d    l %d   md %d   r %d   x %d  y %d   realX %d  realY %d\n",
            //     sorted_k, i, l, md, r, x, y, input_data[realX], input_data[realY]);
        // }
        if (realX < n && realY < n && input_data[realX] > input_data[realY]) {
            l = md;
        } else {
            r = md;
        }
    }

    int x, y;
    pairByDiagAndNum(xStart, yStart, l, &x, &y);
    x -= 1;
    // printf("predok: %d %d\t%d %d\n", x, y, toRealIdxX(x, sorted_k, block), toRealIdxY(y, sorted_k, block));
    if (y >= blockStart && 
        (x < blockStart || (input_data[toRealIdxX(x, sorted_k, block)] <= input_data[toRealIdxY(y, sorted_k, block)]))) {
        ++x;
    } else {
        ++y;
    }
    
    const int realX = toRealIdxX(x, sorted_k, block);
    const int realY = toRealIdxY(y, sorted_k, block);
    
    const uint inX = ((x - blockStart < sorted_k && realX < n) ? input_data[realX] : UINT_MAX);
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
    // printf("i: %5d\tx: %5d\txStart: %5d\ty: %5d\tyStart: %5d\tblock: %5d\tsorted_k: %5d\n", 
    //     i, x, xStart, y, yStart, block, sorted_k);
    // printf("iInBlock: %5d\tl: %5d\tr: %5d\tmx: %5d\n", 
    //     iInBlock, l, r, mx);
    // printf("blockXStart: %5d\t blockYStart: %5d\n",
    //     blockXStart, blockYStart);
    // rassert(x + y < n, 40973615);
    // rassert(inIdx < n, 765237465);
    // printf("i: %d\tblock: %d\t x: %5d\ty: %5d\tinX: %d\tinY: %d\tout: %d\nl: %d\tr %d\tmx: %d\nblockStart: %d\txStart: %d\tyStart %d\n\n", 
    //     i, block, x, y, inX, inY, out,
    //     l, r, mx,
    //     blockStart, xStart, yStart);
    if (inIdx < sorted_k){
        output_data[x + y] = out;
    }
}
