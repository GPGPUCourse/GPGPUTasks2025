#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/matrix_vector_multiply.h"

namespace ocl {
const ocl::ProgramBinaries& getAplusB()
{
    return opencl_binaries_aplusb;
}

const ProgramBinaries& getMatrixVectorMult()
{
    return opencl_binaries_matrix_vector_multiply;
}
} // namespace ocl
