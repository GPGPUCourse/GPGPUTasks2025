#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize& workSize, gpu::gpu_mem_32u& buffer, unsigned int n);
void radix_sort_01_local_counting(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& arr_in, gpu::gpu_mem_32u& buffer, unsigned int n, unsigned int offset);
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& pow2_sum, gpu::gpu_mem_32u& next_pow2_sum, unsigned int n);
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& pow2_sum, gpu::gpu_mem_32u& prefix_sum_accum, unsigned int n, unsigned int pow2);
void radix_sort_04_scatter(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& arr_in, const gpu::gpu_mem_32u& pref_sum, gpu::gpu_mem_32u& arr_out, unsigned int n, unsigned int offset);
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