#include "utils.h"

#include "context.h"

#include <libgpu/opencl/engine.h>
#include <limits>


void gpu::raiseException(std::string file, int line, std::string message)  {
	if (message.length() > 0) {
		throw gpu_exception("Failure at " + file + ":" + to_string(line) + ": " + message);
	} else {
		throw gpu_exception("Failure at " + file + ":" + to_string(line));
	}
}

template <typename T>
size_t gpu::deviceTypeSize() {
	Context context;
#ifdef CUDA_SUPPORT
	if (context.type() == Context::TypeCUDA) {
		return sizeof(T);
	} else
#endif
	if (context.type() == Context::TypeOpenCL) {
		return sizeof(typename ocl::OpenCLType<T>::type);
	} else {
		throw gpu_exception("No GPU active context");
	}
}

template <typename T>
T gpu::deviceTypeMax() {
	Context context;
#ifdef CUDA_SUPPORT
	if (context.type() == Context::TypeCUDA) {
		return std::numeric_limits<T>::max();
	} else
#endif
	if (context.type() == Context::TypeOpenCL) {
		return ocl::OpenCLType<T>::max();
	} else {
		throw gpu_exception("No GPU active context");
	}
}

template <typename T>
T gpu::deviceTypeMin() {
	Context context;
#ifdef CUDA_SUPPORT
	if (context.type() == Context::TypeCUDA) {
		return std::numeric_limits<T>::min();
	} else
#endif
	if (context.type() == Context::TypeOpenCL) {
		return ocl::OpenCLType<T>::min();
	} else {
		throw gpu_exception("No GPU active context");
	}
}

unsigned int gpu::calcNChunk(size_t n, size_t group_size, size_t max_size)
{
	if (n == 0)
		return (unsigned int) group_size;

	size_t work_parts_n = (n + max_size - 1) / max_size;
	size_t exec_n = (n + work_parts_n - 1) / work_parts_n;
	exec_n = (exec_n + group_size - 1) / group_size * group_size;
	return (unsigned int) exec_n;
}

unsigned int gpu::calcColsChunk(size_t width, size_t height, size_t group_size_x, size_t max_size)
{
	if (width == 0 || height == 0)
		return (unsigned int) group_size_x;

	size_t work_parts_n = (width * height + max_size - 1) / max_size;
	size_t ncols = (width + work_parts_n - 1) / work_parts_n;
	ncols = (ncols + group_size_x - 1) / group_size_x * group_size_x;
	return (unsigned int) ncols;
}

unsigned int gpu::calcRowsChunk(size_t width, size_t height, size_t group_size_y, size_t max_size)
{
	if (width == 0 || height == 0)
		return (unsigned int) group_size_y;

	size_t work_parts_n = (width * height + max_size - 1) / max_size;
	size_t nrows = (height + work_parts_n - 1) / work_parts_n;
	nrows = (nrows + group_size_y - 1) / group_size_y * group_size_y;
	return (unsigned int) nrows;
}

unsigned int gpu::calcZSlicesChunk(size_t x, size_t y, size_t z, size_t group_size_z, size_t max_size)
{
	if (x == 0 || y == 0 || z == 0)
		return (unsigned int) group_size_z;

	size_t work_parts_n = (z * y * x + max_size - 1) / max_size;
	size_t z_slices = (z + work_parts_n - 1) / work_parts_n;
	z_slices = (z_slices + group_size_z - 1) / group_size_z * group_size_z;
	return (unsigned int) z_slices;
}

template size_t		gpu::deviceTypeSize<int8_t>();
template size_t		gpu::deviceTypeSize<int16_t>();
template size_t		gpu::deviceTypeSize<int32_t>();
template size_t		gpu::deviceTypeSize<uint8_t>();
template size_t		gpu::deviceTypeSize<uint16_t>();
template size_t		gpu::deviceTypeSize<uint32_t>();
template size_t		gpu::deviceTypeSize<float>();
template size_t		gpu::deviceTypeSize<double>();

template int8_t		gpu::deviceTypeMax<int8_t>();
template int16_t	gpu::deviceTypeMax<int16_t>();
template int32_t	gpu::deviceTypeMax<int32_t>();
template uint8_t	gpu::deviceTypeMax<uint8_t>();
template uint16_t	gpu::deviceTypeMax<uint16_t>();
template uint32_t	gpu::deviceTypeMax<uint32_t>();
template float		gpu::deviceTypeMax<float>();
template double		gpu::deviceTypeMax<double>();

template int8_t		gpu::deviceTypeMin<int8_t>();
template int16_t	gpu::deviceTypeMin<int16_t>();
template int32_t	gpu::deviceTypeMin<int32_t>();
template uint8_t	gpu::deviceTypeMin<uint8_t>();
template uint16_t	gpu::deviceTypeMin<uint16_t>();
template uint32_t	gpu::deviceTypeMin<uint32_t>();
template float		gpu::deviceTypeMin<float>();
template double		gpu::deviceTypeMin<double>();
