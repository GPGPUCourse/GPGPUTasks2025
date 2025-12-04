#include "aplusb_matrix_good_nospir_opencl120.h"

namespace ocl {
	static const ocl::VersionedBinary* opencl_versioned_binaries_aplusb_matrix_good[] = {
		&opencl_binary_aplusb_matrix_good_nospir_opencl120
	};
	static ProgramBinaries opencl_binaries_aplusb_matrix_good = ProgramBinaries(std::vector<const VersionedBinary *>(opencl_versioned_binaries_aplusb_matrix_good, opencl_versioned_binaries_aplusb_matrix_good + sizeof(opencl_versioned_binaries_aplusb_matrix_good) / sizeof(opencl_versioned_binaries_aplusb_matrix_good[0])), "aplusb_matrix_good");
}
