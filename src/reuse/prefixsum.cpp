#include "prefixsum.h"

#include "defines.h"
#include "wrappers.h"
#include "debug.h"

void calcPrefixSum(gpuptr::u32 a, gpuptr::u32 b, gpuptr::u32 c, unsigned int n,
                   unsigned int depth) {
    int bsz = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    if (n <= GROUP_SIZE) {  // trivial
        cuda::prefixsum_main(gpu::WorkSize(GROUP_SIZE, n), a, c, n);
        return;
    }
    auto buf = b.allocate(bsz);
    cuda::prefixsum_pre(gpu::WorkSize(GROUP_SIZE, n), a, buf, c, n);
    printVec("prefsum pre buf", buf, bsz, "\n");
    // need to calculate pref for b[bbase;bbase+bsz)
    calcPrefixSum(buf,        // start is b[bbase]
                  b,  // buffer starts at b[bbase+bsz]
                  buf,        // calulate prefsum inplace
                  bsz, depth + 1);
    printVec("prefsum aster rec buf", buf, bsz, "\n");


    cuda::prefixsum_post(gpu::WorkSize(GROUP_SIZE, n), buf, c, n);
}

void prefixSum(gpuptr::u32 in, gpuptr::u32 out, gpuptr::u32 buffer) {
    calcPrefixSum(in, buffer, out, in.size(), 0);
}