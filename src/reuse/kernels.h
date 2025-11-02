#pragma once

#include "wrappers.h"

namespace cuda {

void copy(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 b, unsigned int n);
void fill(const gpu::WorkSize& workSize, gpuptr::u32 b, unsigned int x, unsigned int n);
void prefixsum_pre(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 b, gpuptr::u32 c, unsigned int n);
void prefixsum_main(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 c, unsigned int n);
void prefixsum_post(const gpu::WorkSize& workSize, gpuptr::u32 b, gpuptr::u32 c, unsigned int n);

void radix_pre(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 b, unsigned int offset, unsigned int n);
void radix_post(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 b, gpuptr::u32 c, unsigned int offset, unsigned int n);

void mergesort(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 c, unsigned int size, unsigned int n);
}