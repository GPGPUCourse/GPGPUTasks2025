#pragma once

#include <libgpu/vulkan/engine.h>

namespace cpu {
void mandelbrot(float* results,
                unsigned int width, unsigned int height,
                float fromX, float fromY,
                float sizeX, float sizeY,
                unsigned int iters, unsigned int isSmoothing, bool useOpenMP);

unsigned int sum(const unsigned int *values, unsigned int n);
unsigned int sumOpenMP(const unsigned int *values, unsigned int n);
}

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);

void mandelbrot(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &results,
    unsigned int width, unsigned int height,
    float fromX, float fromY,
    float sizeX, float sizeY,
    unsigned int iters, unsigned int isSmoothing);

void sum_01_atomics(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, gpu::gpu_mem_32u& sum, unsigned int n);
void sum_02_atomics_load_k(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, gpu::gpu_mem_32u& sum, unsigned int n);
void sum_03_local_memory_atomic_per_workgroup(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, gpu::gpu_mem_32u& sum, unsigned int n);
void sum_04_local_reduction(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, gpu::gpu_mem_32u& b, unsigned int n);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getMandelbrot();

const ProgramBinaries& getSum01Atomics();
const ProgramBinaries& getSum02AtomicsLoadK();
const ProgramBinaries& getSum03LocalMemoryAtomicPerWorkgroup();
const ProgramBinaries& getSum04LocalReduction();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getMandelbrot();

const ProgramBinaries& getSum01Atomics();
const ProgramBinaries& getSum02AtomicsLoadK();
const ProgramBinaries& getSum03LocalMemoryAtomicPerWorkgroup();
const ProgramBinaries& getSum04LocalReduction();
}
