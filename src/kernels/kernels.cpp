#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/fill.h"
#include "cl/generated_kernels/radix_sort_onehot.h"
#include "cl/generated_kernels/radix_sort_prefix_sum_accumulation.h"
#include "cl/generated_kernels/radix_sort_prefix_sum_reduction.h"
#include "cl/generated_kernels/radix_sort_scatter.h"

namespace ocl {
const ocl::ProgramBinaries& getAplusB() {
    return opencl_binaries_aplusb;
}

const ProgramBinaries& getFill() {
    return opencl_binaries_fill;
}

const ProgramBinaries& getRadixSortOnehot() {
    return opencl_binaries_radix_sort_onehot;
}

const ProgramBinaries& getRadixSortPrefixSumAccumulation() {
    return opencl_binaries_radix_sort_prefix_sum_accumulation;
}

const ProgramBinaries& getRadixSortPrefixSumReduction() {
    return opencl_binaries_radix_sort_prefix_sum_reduction;
}

const ProgramBinaries& getRadixSortScatter() {
    return opencl_binaries_radix_sort_scatter;
}
} // namespace ocl
