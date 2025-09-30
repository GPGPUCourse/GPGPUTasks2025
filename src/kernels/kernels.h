#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void aplusb_matrix_bad(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int width, unsigned int height);
void aplusb_matrix_good(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int width, unsigned int height);
}

namespace ocl {
const ProgramBinaries& getAplusB();
const ProgramBinaries& getAplusBMatrixBad();
const ProgramBinaries& getAplusBMatrixGood();
}

namespace avk2 {
const ProgramBinaries& getAplusB();
const ProgramBinaries& getAplusBMatrixBad();
const ProgramBinaries& getAplusBMatrixGood();
}
