#ifdef CUDA_SUPPORT
#include "cuda_api.h"

#ifdef _WIN32

#include <windows.h>

typedef HMODULE CudaLibrary;

static HMODULE cudaLoadLibrary()
{
	return LoadLibraryW(L"nvcuda.dll");
}

static FARPROC cudaGetProcAddress(HMODULE hModule, LPCSTR lpProcName)
{
	return ::GetProcAddress(hModule, lpProcName);
}

#elif defined(__unix__) || defined(__APPLE__) || defined(__MACOSX)

#include <dlfcn.h>

typedef void * CudaLibrary;

static CudaLibrary cudaLoadLibrary()
{
#if defined(__APPLE__) || defined(__MACOSX)
	return dlopen("/Library/Frameworks/CUDA.framework/Versions/Current/CUDA", RTLD_NOW);
#else
	CudaLibrary lib = dlopen("libcuda.so", RTLD_NOW);
	if (!lib) {
		lib = dlopen("libcuda.so.1", RTLD_NOW);
	}
	return lib;
#endif
}

static void *cudaGetProcAddress(void *handle, const char *symbol)
{
	return dlsym(handle, symbol);
}

#else
#error unsupported platform
#endif

namespace cuda {

std::string driverErrorString(CUresult code)
{
#define DEFINE_ERROR(value)	case value: return #value;

	switch (code) {
	DEFINE_ERROR(CUDA_SUCCESS)
	DEFINE_ERROR(CUDA_ERROR_INVALID_VALUE)
	DEFINE_ERROR(CUDA_ERROR_OUT_OF_MEMORY)
	DEFINE_ERROR(CUDA_ERROR_NOT_INITIALIZED)
	DEFINE_ERROR(CUDA_ERROR_DEINITIALIZED)
	DEFINE_ERROR(CUDA_ERROR_PROFILER_DISABLED)
	DEFINE_ERROR(CUDA_ERROR_PROFILER_NOT_INITIALIZED)
	DEFINE_ERROR(CUDA_ERROR_PROFILER_ALREADY_STARTED)
	DEFINE_ERROR(CUDA_ERROR_PROFILER_ALREADY_STOPPED)
	DEFINE_ERROR(CUDA_ERROR_NO_DEVICE)
	DEFINE_ERROR(CUDA_ERROR_INVALID_DEVICE)
	DEFINE_ERROR(CUDA_ERROR_INVALID_IMAGE)
	DEFINE_ERROR(CUDA_ERROR_INVALID_CONTEXT)
	DEFINE_ERROR(CUDA_ERROR_CONTEXT_ALREADY_CURRENT)
	DEFINE_ERROR(CUDA_ERROR_MAP_FAILED)
	DEFINE_ERROR(CUDA_ERROR_UNMAP_FAILED)
	DEFINE_ERROR(CUDA_ERROR_ARRAY_IS_MAPPED)
	DEFINE_ERROR(CUDA_ERROR_ALREADY_MAPPED)
	DEFINE_ERROR(CUDA_ERROR_NO_BINARY_FOR_GPU)
	DEFINE_ERROR(CUDA_ERROR_ALREADY_ACQUIRED)
	DEFINE_ERROR(CUDA_ERROR_NOT_MAPPED)
	DEFINE_ERROR(CUDA_ERROR_NOT_MAPPED_AS_ARRAY)
	DEFINE_ERROR(CUDA_ERROR_NOT_MAPPED_AS_POINTER)
	DEFINE_ERROR(CUDA_ERROR_ECC_UNCORRECTABLE)
	DEFINE_ERROR(CUDA_ERROR_UNSUPPORTED_LIMIT)
	DEFINE_ERROR(CUDA_ERROR_CONTEXT_ALREADY_IN_USE)
	DEFINE_ERROR(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED)
	DEFINE_ERROR(CUDA_ERROR_INVALID_SOURCE)
	DEFINE_ERROR(CUDA_ERROR_FILE_NOT_FOUND)
	DEFINE_ERROR(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND)
	DEFINE_ERROR(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED)
	DEFINE_ERROR(CUDA_ERROR_OPERATING_SYSTEM)
	DEFINE_ERROR(CUDA_ERROR_INVALID_HANDLE)
	DEFINE_ERROR(CUDA_ERROR_NOT_FOUND)
	DEFINE_ERROR(CUDA_ERROR_NOT_READY)
	DEFINE_ERROR(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES)
	DEFINE_ERROR(CUDA_ERROR_LAUNCH_TIMEOUT)
	DEFINE_ERROR(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING)
	DEFINE_ERROR(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
	DEFINE_ERROR(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED)
	DEFINE_ERROR(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE)
	DEFINE_ERROR(CUDA_ERROR_CONTEXT_IS_DESTROYED)
	DEFINE_ERROR(CUDA_ERROR_ASSERT)
	DEFINE_ERROR(CUDA_ERROR_TOO_MANY_PEERS)
	DEFINE_ERROR(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
	DEFINE_ERROR(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED)
	DEFINE_ERROR(CUDA_ERROR_LAUNCH_FAILED)
	DEFINE_ERROR(CUDA_ERROR_NOT_PERMITTED)
	DEFINE_ERROR(CUDA_ERROR_NOT_SUPPORTED)
	DEFINE_ERROR(CUDA_ERROR_UNKNOWN)
	default: return "CUDA_ERROR_UNKNOWN_CODE_" + to_string(code);
	}

#undef DEFINE_ERROR
}

std::string formatDriverError(CUresult code)
{
	return driverErrorString(code) + " (" + to_string(code) + ")";
}

}

typedef CUresult				(CUDAAPI * p_pfn_cuDeviceGet)				(CUdevice *, int);
typedef CUresult				(CUDAAPI * p_pfn_cuCtxCreate)				(CUcontext *, unsigned int, CUdevice);
typedef CUresult				(CUDAAPI * p_pfn_cuCtxDestroy)				(CUcontext);

p_pfn_cuDeviceGet				pfn_cuDeviceGet				= 0;
p_pfn_cuCtxCreate				pfn_cuCtxCreate				= 0;
p_pfn_cuCtxDestroy				pfn_cuCtxDestroy			= 0;

bool cuda_api_init()
{
	if (pfn_cuCtxCreate)
		return true;

	CudaLibrary lib = cudaLoadLibrary();
	if (!lib)
		return false;

	pfn_cuDeviceGet				= (p_pfn_cuDeviceGet)				cudaGetProcAddress(lib, "cuDeviceGet");
	pfn_cuCtxCreate				= (p_pfn_cuCtxCreate)				cudaGetProcAddress(lib, "cuCtxCreate_v2");
	pfn_cuCtxDestroy			= (p_pfn_cuCtxDestroy)				cudaGetProcAddress(lib, "cuCtxDestroy_v2");

	return true;
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal)
{
	if (!pfn_cuDeviceGet) return CUDA_ERROR_NOT_INITIALIZED;

	return pfn_cuDeviceGet(device, ordinal);
}

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
	if (!pfn_cuCtxCreate) return CUDA_ERROR_NOT_INITIALIZED;

	return pfn_cuCtxCreate(pctx, flags, dev);
}

CUresult CUDAAPI cuCtxDestroy(CUcontext ctx)
{
	if (!pfn_cuCtxDestroy) return CUDA_ERROR_NOT_INITIALIZED;

	return pfn_cuCtxDestroy(ctx);
}
#endif
