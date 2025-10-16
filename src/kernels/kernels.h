#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);

void prefix_sum_prepare(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &a, const unsigned int abase,
    gpu::gpu_mem_32u &b, const unsigned int bbase,
    gpu::gpu_mem_32u &c, const unsigned int cbase,
    unsigned int n);

void prefix_sum_simple(const gpu::WorkSize &workSize, 
    const gpu::gpu_mem_32u &a, unsigned int abase,
    gpu::gpu_mem_32u &c, unsigned int cbase,
    unsigned int n);

void prefix_sum_post(const gpu::WorkSize &workSize,
    gpu::gpu_mem_32u &b, const unsigned int bbase,
    gpu::gpu_mem_32u &c, const unsigned int cbase,
    unsigned int n);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getPrefixSum01Reduction();
const ProgramBinaries& getPrefixSum02PrefixAccumulation();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getPrefixSum01Reduction();
const ProgramBinaries& getPrefixSum02PrefixAccumulation();
}
