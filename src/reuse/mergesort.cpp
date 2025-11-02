#include "mergesort.h"

#include "defines.h"
#include "wrappers.h"
#include "debug.h"


void mergeSort(gpuptr::u32 in, gpuptr::u32 out, gpuptr::u32 buffer, int max_val) {
    int n = in.size();
    for (int i = 0; i < 32; ++i) {
        if ((1 << i) >= n) {
            n = (1 << i);
            break;
        }
    }
    gpuptr::u32 bufs[2] = {buffer.allocate(n), buffer.allocate(n)};
    printVec("in", in, in.size(), "\n");

    cuda::fill(gpu::WorkSize(GROUP_SIZE, n), bufs[0], max_val, n);
    cuda::copy(gpu::WorkSize(GROUP_SIZE, in.size()), in, bufs[0], in.size());
    int i = 0;
    for (; (1 << i) < n; ++i) {
        auto& l = bufs[i%2];
        auto& r = bufs[1-i%2];
        printVec("bufs[l]", l, n, "\n");
        cuda::mergesort(gpu::WorkSize(GROUP_SIZE, n), l, r, (1 << i), n);
        printVec("bufs[r]", r, n, "\n");
        
    }
    cuda::copy(gpu::WorkSize(GROUP_SIZE, in.size()), bufs[i%2], out, in.size());  
}