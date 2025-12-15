#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);
void radix_sort_01_local_counting(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &values, gpu::gpu_mem_32u &buffer1, unsigned int a1, unsigned int a2);
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1);
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2);
void radix_sort_04_scatter(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2);

// Антихайп
void RadixSortLocalCount0(
    const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1,
    gpu::gpu_mem_32u &buffer2,
    unsigned int a1,
    unsigned int a2
);

void RadixSortLocalCount1(
    const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1,
    gpu::gpu_mem_32u &buffer2,
    unsigned int a1,
    unsigned int a2
);

void InclusiveScan(
    const gpu::WorkSize& ws,
    gpu::gpu_mem_32u& input_gpu,
    gpu::gpu_mem_32u& buffer1_pow2_sum_gpu,
    gpu::gpu_mem_32u& buffer2_pow2_sum_gpu,
    gpu::gpu_mem_32u& prefix_sum_accum_gpu,
    unsigned int n
);

void RadixSortScatter(
    const gpu::WorkSize& workSize,
    gpu::gpu_mem_32u& input_buf,
    gpu::gpu_mem_32u& counted_buf,
    gpu::gpu_mem_32u& preffix_sums_buf0,
    gpu::gpu_mem_32u& preffix_sums_buf1,
    gpu::gpu_mem_32u& output_buf,
    unsigned int n,
    unsigned int offset);

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
