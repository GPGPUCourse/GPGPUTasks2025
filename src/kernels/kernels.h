#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void merge_sort(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getMergeSort();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getMergeSort();
}
