#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
void radix_sort_01_scan_build_fenwick(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &output, unsigned int n, unsigned int d, unsigned int bit);
void radix_sort_02_scan_accumulation(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &buffer_fenwick_gpu, gpu::gpu_mem_32u &prefix_sum_accum_gpu, unsigned int n);
void radix_sort_03_scatter(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &input, const gpu::gpu_mem_32u &scan, gpu::gpu_mem_32u &output, unsigned int n, unsigned int bit, unsigned int digit);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getRadixSort01LocalCounting();
const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReduction();
const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulation();
const ProgramBinaries& getRadixSort04Scatter();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getRadixSort01LocalCounting();
const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReduction();
const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulation();
const ProgramBinaries& getRadixSort04Scatter();
}
