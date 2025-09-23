#pragma once

#include "exceptions.h"
#include <libgpu/utils.h>
#include "libbase/string_utils.h"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <iostream>
#include <cfloat>

namespace cuda {

	std::string formatError(cudaError_t code);

	void reportError(cudaError_t err, int line, const std::string &prefix = std::string());

	void checkKernelErrors(cudaStream_t stream, int line);
	void checkKernelErrors(cudaStream_t stream, int line, bool synchronized);

	#define CUDA_SAFE_CALL(expr) cuda::reportError(expr, __LINE__)
	#define CUDA_TRACE(expr) expr;
	#define CUDA_CHECK_KERNEL(stream) cuda::checkKernelErrors(stream, __LINE__)
	#define CUDA_CHECK_KERNEL_SYNC(stream) cuda::checkKernelErrors(stream, __LINE__, true)
	#define CUDA_CHECK_KERNEL_ASYNC(stream) cuda::checkKernelErrors(stream, __LINE__, false)

	template <typename T>	class DataTypeRange					{ };
	template<>				class DataTypeRange<unsigned char>	{ public:	static __device__	unsigned char	min() { return 0; }			static __device__	unsigned char	max() {	return UCHAR_MAX;	}};
	template<>				class DataTypeRange<unsigned short>	{ public:	static __device__	unsigned short	min() { return 0; }			static __device__	unsigned short	max() {	return USHRT_MAX;	}};
	template<>				class DataTypeRange<unsigned int>	{ public:	static __device__	unsigned int	min() { return 0; }			static __device__	unsigned int	max() {	return UINT_MAX;	}};
	template<>				class DataTypeRange<float>			{ public:	static __device__	float			min() { return FLT_MIN; }	static __device__	float			max() {	return FLT_MAX;		}};
	template<>				class DataTypeRange<double>			{ public:	static __device__	double			min() { return DBL_MIN; }	static __device__	double			max() {	return DBL_MAX;		}};

	template <typename T>	class TypeHelper					{ };
	template<>				class TypeHelper<unsigned char>		{ public:	typedef unsigned int	type32; };
	template<>				class TypeHelper<unsigned short>	{ public:	typedef unsigned int	type32; };
	template<>				class TypeHelper<unsigned int>		{ public:	typedef unsigned int	type32; };
	template<>				class TypeHelper<float>				{ public:	typedef float			type32; };
	template<>				class TypeHelper<double>			{ public:	typedef float			type32; };

}
