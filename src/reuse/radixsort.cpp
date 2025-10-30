
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
    // cuda::copy(gpu::WorkSize(GROUP_SIZE, n), in, bufs[0], n);
    for (int i = 0; i < 32; ++i) {
        auto l = &bufs[i & 1];
        auto r = &bufs[i & 1 ^ 1];
        if (i == 0) {
            l = &in;
        }
        if (i == 31) {
            r = &out;
        }
        // printVecBin("bufs[l]", l, n, "\n");
        cuda::radix_pre(gpu::WorkSize(GROUP_SIZE, n), *l, bbuf, i, n);
        // printVec("bbuf", bbuf, n, "\n");
        prefixSum(bbuf, bbufpref, prefixbuf);
        // printVec("bbuf[pref]", bbufpref, n, "\n");
        cuda::radix_post(gpu::WorkSize(GROUP_SIZE, n), *l, bbufpref, *r, i, n);
        // printVecBin("bufs[r]", r, n, "\n");
    }
    // cuda::copy(gpu::WorkSize(GROUP_SIZE, n), bufs[0], out, n);
}
