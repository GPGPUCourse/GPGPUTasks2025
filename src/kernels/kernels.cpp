#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/fill.h"
#include "cl/generated_kernels/radix_sort_onehot.h"
#include "cl/generated_kernels/prefix_sum_accumulation.h"
#include "cl/generated_kernels/prefix_sum_reduction.h"
#include "cl/generated_kernels/radix_sort_scatter.h"
#include "cl/generated_kernels/hillis_steele.h"


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

const ProgramBinaries& getPrefixSumAccumulation() {
    return opencl_binaries_prefix_sum_accumulation;
}

const ProgramBinaries& getPrefixSumReduction() {
    return opencl_binaries_prefix_sum_reduction;
}

const ProgramBinaries& getHillisSteele() {
    return opencl_binaries_hillis_steele;
}

const ProgramBinaries& getRadixSortScatter() {
    return opencl_binaries_radix_sort_scatter;
}
} // namespace ocl
