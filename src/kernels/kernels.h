#pragma once

#include <libgpu/vulkan/engine.h>

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFill();
const ProgramBinaries& getRadixSortOnehot();
const ProgramBinaries& getPrefixSumAccumulation();
const ProgramBinaries& getPrefixSumReduction();
const ProgramBinaries& getHillisSteele();
const ProgramBinaries& getRadixSortScatter();
}
