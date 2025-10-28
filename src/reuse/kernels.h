#pragma once

#include "wrappers.h"

namespace cuda {

void prefixsum_pre(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 b, gpuptr::u32 c, unsigned int n);
void prefixsum_main(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 c, unsigned int n);
void prefixsum_post(const gpu::WorkSize& workSize, gpuptr::u32 b, gpuptr::u32 c, unsigned int n);

}