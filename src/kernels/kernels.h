#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize& workSize, gpu::gpu_mem_32u& buffer, unsigned int n);
void radix_sort_01_local_counting(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& buffer1,
    gpu::gpu_mem_32u& buffer2,
    unsigned int bitCount,
    unsigned int offset,
    unsigned int n);
void radix_sort_02_global_prefixes_scan_sum_reduction(
    const gpu::gpu_mem_32u& buffer1,
    gpu::gpu_mem_32u& buffer2,
    unsigned int n,
    unsigned int bitCount,
    unsigned int offset,
    unsigned int mask);
void radix_sort_03_global_prefixes_scan_accumulation(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& buffer1,
    const gpu::gpu_mem_32u& global_offsets,
    gpu::gpu_mem_32u& buffer2,
    unsigned int n);
void radix_sort_04_scatter(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_gpu,
    const gpu::gpu_mem_32u& scan_buffer_gpu,
    gpu::gpu_mem_32u& output_gpu,
    unsigned int bitCount,
    unsigned int offset,
    unsigned int local_offset,
    unsigned int n);
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
