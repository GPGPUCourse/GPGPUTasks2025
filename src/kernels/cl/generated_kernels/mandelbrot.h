#include "mandelbrot_nospir_opencl120.h"

namespace ocl {
	static const ocl::VersionedBinary* opencl_versioned_binaries_mandelbrot[] = {
		&opencl_binary_mandelbrot_nospir_opencl120
	};
	static ProgramBinaries opencl_binaries_mandelbrot = ProgramBinaries(std::vector<const VersionedBinary *>(opencl_versioned_binaries_mandelbrot, opencl_versioned_binaries_mandelbrot + sizeof(opencl_versioned_binaries_mandelbrot) / sizeof(opencl_versioned_binaries_mandelbrot[0])), "mandelbrot");
}
