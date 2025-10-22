#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
void prefix_sum_01_sum_reduction(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &next_pow2_sum, unsigned int n);
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n, unsigned int pow2);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getPrefixSum01Reduction();
const ProgramBinaries& getPrefixSum02PrefixAccumulation();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getPrefixSum01Reduction();
const ProgramBinaries& getPrefixSum02PrefixAccumulation();
}
