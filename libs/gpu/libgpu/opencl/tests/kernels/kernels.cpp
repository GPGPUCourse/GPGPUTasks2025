#include "kernels.h"

#include "generated_kernels/aplusb.h"

namespace ocl {
	ProgramBinaries& getAplusBKernel() {
		return opencl_binaries_aplusb;
	}
}
