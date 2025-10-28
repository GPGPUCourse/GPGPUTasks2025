
#include "defines.h"
#include "kernels.h"
#include "prefixsum.h"
#include "wrappers.h"
#include "debug.h"
#include <cstdio>



void radixSort(gpuptr::u32 in, gpuptr::u32 out, gpuptr::u32 buf) {
    unsigned int n = in.size();
    gpuptr::u32 bufs[2] = {buf.allocate(n), buf.allocate(n)};
    gpuptr::u32 bbuf = buf.allocate(n);
    gpuptr::u32 bbufpref = buf.allocate(n);
    gpuptr::u32 prefixbuf = buf.allocate(2 * n);
    cuda::copy(gpu::WorkSize(GROUP_SIZE, n), in, bufs[0], n);
    for (int i = 0; i < 32; ++i) {
        auto& l = bufs[i % 2];
        auto& r = bufs[1 - (i % 2)];
        printVecBin("bufs[l]", l, n, "\n");
        cuda::radix_pre(gpu::WorkSize(GROUP_SIZE, n), l, bbuf, i, n);
        printVec("bbuf", bbuf, n, "\n");
        prefixSum(bbuf, bbufpref, prefixbuf);
        printVec("bbuf[pref]", bbufpref, n, "\n");
        unsigned int sum = bbufpref.at(n - 1);
        print("sum=%d\n", sum);
    
        cuda::radix_post(gpu::WorkSize(GROUP_SIZE, n), l, bbufpref, r, sum, i, n);
        printVecBin("bufs[r]", r, n, "\n");


    }
    cuda::copy(gpu::WorkSize(GROUP_SIZE, n), bufs[0], out, n);
}
