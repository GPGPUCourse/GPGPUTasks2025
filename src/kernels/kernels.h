#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
void radix_sort_01_local_counting(const gpu::gpu_mem_32u& in, gpu::gpu_mem_32u& block_hist, const unsigned int& n, const unsigned int& shift, const unsigned int& blocks_cnt);
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::gpu_mem_32u& block_hist, gpu::gpu_mem_32u& block_offsets, gpu::gpu_mem_32u& bin_counter, const unsigned int& blocks_cnt);
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::gpu_mem_32u& bin_counter, gpu::gpu_mem_32u& bin_base);
void radix_sort_04_scatter(const gpu::gpu_mem_32u& in, const gpu::gpu_mem_32u& block_offsets, const gpu::gpu_mem_32u& bin_base, gpu::gpu_mem_32u& out, const unsigned int& n, const unsigned int& shift, const unsigned int& blocks_cnt);
void prefix_sum(const unsigned int* in, unsigned int* out, const unsigned int& n);
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
