#pragma once

#include <libgpu/vulkan/engine.h>

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFill();
const ProgramBinaries& getRadixSortOnehot();
const ProgramBinaries& getRadixSortPrefixSumAccumulation();
const ProgramBinaries& getRadixSortPrefixSumReduction();
const ProgramBinaries& getRadixSortScatter();
}
