#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
void radix_sort_01_map(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& values, gpu::gpu_mem_32u& buffer1, unsigned int a1, unsigned int a2);
void radix_sort_02_scan_sum_reduction(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& buffer1, gpu::gpu_mem_32u& buffer2, unsigned int a1, unsigned int a2);
void radix_sort_03_prefix_accumulation(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& buffer1, gpu::gpu_mem_32u& buffer2, unsigned int a1, unsigned int a2);
void radix_sort_04_scatter(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& values, const gpu::gpu_mem_32u& buffer1, gpu::gpu_mem_32u& buffer2, gpu::gpu_mem_32u& buffer3, unsigned int a1);
void copy_buffer(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& from, gpu::gpu_mem_32u& to, unsigned int n);
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
