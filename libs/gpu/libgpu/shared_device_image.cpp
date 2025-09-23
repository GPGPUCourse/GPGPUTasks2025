#define _SHORT_FILE_ "shared_device_image.cpp"

#include "shared_device_image.h"
#include "context.h"
#include <libbase/runtime_assert.h>
#include <libbase/string_utils.h>
#include <libgpu/utils.h>
#include <algorithm>
#include <stdexcept>
#include <cstring>

#ifdef CUDA_SUPPORT
#include <libgpu/cuda/utils.h>
#include <cuda_runtime.h>
#endif

#include "vulkan/engine.h"
#include "vulkan/vulkan_api_headers.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace gpu {

shared_device_image::shared_device_image()
{
	buffer_		= 0;
	data_		= 0;
	cu_texture_	= 0;
	cu_surface_	= 0;
	type_		= Context::TypeUndefined;

	width_		= 0;
	height_		= 0;
	cn_			= 0;
	data_type_	= DataTypeUndefined;
}

shared_device_image::~shared_device_image()
{
	try {
		decref();
	}
	catch (std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
	}
}

shared_device_image::shared_device_image(const shared_device_image &other)
{
	buffer_		= other.buffer_;
	data_		= other.data_;
	cu_texture_	= other.cu_texture_;
	cu_surface_	= other.cu_surface_;
	type_		= other.type_;

	width_		= other.width_;
	height_		= other.height_;
	cn_			= other.cn_;
	data_type_	= other.data_type_;
	incref();
}

shared_device_image &shared_device_image::operator= (const shared_device_image &other)
{
	if (this != &other) {
		decref();
		buffer_		= other.buffer_;
		data_		= other.data_;
		cu_texture_	= other.cu_texture_;
		cu_surface_	= other.cu_surface_;
		type_		= other.type_;

		width_		= other.width_;
		height_		= other.height_;
		cn_			= other.cn_;
		data_type_	= other.data_type_;
		incref();
	}

	return *this;
}

void shared_device_image::swap(shared_device_image &other)
{
	std::swap(buffer_,		other.buffer_);
	std::swap(data_,		other.data_);
	std::swap(cu_texture_,	other.cu_texture_);
	std::swap(cu_surface_,	other.cu_surface_);
	std::swap(type_,		other.type_);

	std::swap(width_,		other.width_);
	std::swap(height_,		other.height_);
	std::swap(cn_,			other.cn_);
	std::swap(data_type_,	other.data_type_);
}

void shared_device_image::incref()
{
	if (!buffer_)
		return;

#if defined(_WIN64)
	InterlockedIncrement64((LONGLONG *) buffer_);
#elif defined(_WIN32)
	InterlockedIncrement((LONG *) buffer_);
#else
	__sync_add_and_fetch((long long *) buffer_, 1);
#endif
}

void shared_device_image::decref()
{
	if (!buffer_)
		return;

	long long count = 0;

#if defined(_WIN64)
	count = InterlockedDecrement64((LONGLONG *) buffer_);
#elif defined(_WIN32)
	count = InterlockedDecrement((LONG *) buffer_);
#else
	count = __sync_sub_and_fetch((long long *) buffer_, 1);
#endif

	if (!count) {
		switch (type_) {
#ifdef CUDA_SUPPORT
		case Context::TypeCUDA:
			CUDA_SAFE_CALL(cudaDestroySurfaceObject(cu_surface_));
			CUDA_SAFE_CALL(cudaDestroyTextureObject(cu_texture_));
			CUDA_SAFE_CALL(cudaFreeArray((cudaArray_t) data_));
			break;
#endif
		case Context::TypeOpenCL:
			OCL_SAFE_CALL(clReleaseMemObject((cl_mem) data_));
			break;
		case Context::TypeVulkan:
			delete vkImageData();
			break;
		default:
			gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
		}

		delete [] buffer_;
	}

	buffer_ 	= 0;
	data_		= 0;
	cu_texture_	= 0;
	cu_surface_	= 0;
	type_		= Context::TypeUndefined;

	width_		= 0;
	height_		= 0;
	cn_			= 0;
	data_type_	= DataTypeUndefined;
}

shared_device_image shared_device_image::create(size_t width, size_t height, size_t cn, DataType data_type)
{
	shared_device_image res;
	res.resize(width, height, cn, data_type);
	return res;
}

cudaSurfaceObject_t shared_device_image::cuSurface() const
{
	if (type_ == Context::TypeOpenCL)
		throw gpu_exception("GPU buffer type mismatch");

	return cu_surface_;
}

cudaTextureObject_t shared_device_image::cuTexture() const
{
	if (type_ == Context::TypeOpenCL)
		throw gpu_exception("GPU buffer type mismatch");

	return cu_texture_;
}

cl_mem shared_device_image::clmem() const
{
	if (type_ == Context::TypeCUDA)
		throw gpu_exception("GPU buffer type mismatch");

	return (cl_mem) data_;
}

avk2::raii::ImageData* shared_device_image::vkImageData() const
{
	avk2::raii::ImageData* vk_data = (avk2::raii::ImageData *) data_;
	return vk_data;
}

void shared_device_image::write(const shared_device_buffer &buffer, size_t width, size_t height, size_t src_offset, size_t dst_x_offset, size_t dst_y_offset, bool async)
{
	Context context;
	if (context.type() != type_) {
		gpu::raiseException(_SHORT_FILE_, __LINE__, "Incompatible GPU context");
	}

#ifdef CUDA_SUPPORT
	if (type_ == Context::TypeCUDA) {
		size_t pixel_size = cn() * dataSize(dataType());

		if (async) {
			CUDA_SAFE_CALL(cudaMemcpy2DToArrayAsync((cudaArray_t) data_, dst_x_offset * pixel_size, dst_y_offset, (char*) buffer.cuptr() + src_offset, width * pixel_size, width * pixel_size, height, cudaMemcpyDeviceToDevice, context.cudaStream()));
		} else {
			CUDA_SAFE_CALL(cudaMemcpy2DToArray((cudaArray_t) data_, dst_x_offset * pixel_size, dst_y_offset, (char*) buffer.cuptr() + src_offset, width * pixel_size, width * pixel_size, height, cudaMemcpyDeviceToDevice));
		}
	} else
#endif
	{
		context.cl()->copyBufferToImage(buffer.clmem(), clmem(), width, height, src_offset, dst_x_offset, dst_y_offset, async);
	}
}

void shared_device_image::write(const AnyImage &image)
{
	Context context;
	if (context.type() != type_) {
		gpu::raiseException(_SHORT_FILE_, __LINE__, "Incompatible GPU context");
	}

#ifdef CUDA_SUPPORT
	if (type_ == Context::TypeCUDA) {
		throw std::runtime_error("Unimplemented at line " + to_string(__LINE__));
	} else
#endif
	if (type_ == Context::TypeOpenCL) {
		throw std::runtime_error("Unimplemented at line " + to_string(__LINE__));
	} else if (type_ == Context::TypeVulkan) {
		context.vk()->writeImage(*vkImageData(), image);
	} else {
		rassert(false, 851867772);
	}
}

AnyImage shared_device_image::read() const
{
	AnyImage img(width_, height_, cn_, dataType());

	Context context;
	if (context.type() != type_) {
		gpu::raiseException(_SHORT_FILE_, __LINE__, "Incompatible GPU context");
	}

	size_t row_pitch = img.stride()*dataSize(img.type());

#ifdef CUDA_SUPPORT
	if (type_ == Context::TypeCUDA) {
		size_t pixel_size = cn() * dataSize(dataType());
		size_t x_offset = 0;
		size_t y_offset = 0;

		CUDA_SAFE_CALL(cudaMemcpy2DFromArray(img.ptr(), row_pitch, (cudaArray_t) data_, x_offset*pixel_size, y_offset, width()*pixel_size, height(), cudaMemcpyDeviceToHost));
	} else
#endif
	if (type_ == Context::TypeOpenCL) {
		context.cl()->readImage(clmem(), width(), height(), row_pitch, img.ptr());
	} else if (type_ == Context::TypeVulkan) {
		context.vk()->readImage(*vkImageData(), img);
	} else {
		rassert(false, 851867772);
	}

	return img;
}

AnyImage shared_device_image::read(size_t width, size_t height, size_t offset_x, size_t offset_y, int c) const
{
	rassert(offset_x + width <= width_, 5615228908);
	rassert(offset_y + height <= height_, 32413211321);

	// TODO speedup via loading only cropped data
	AnyImage full_image = read();
	AnyImage cropped_image(full_image, offset_x, offset_y, width, height);

	// TODO speedup shared_device_image::read() via loading only requested channel
	if (c != -1) {
		rassert(c >= 0 && c < cropped_image.channels(), 192268593);
		AnyImage cropped_image_channel(cropped_image.width(), cropped_image.height(), 1, cropped_image.type());
		size_t type_bytes_count = dataSize(cropped_image_channel.type());
		for (ptrdiff_t j = 0; j < cropped_image.height(); ++j) {
			for (ptrdiff_t i = 0; i < cropped_image.width(); ++i) {
				auto src = ((const unsigned char*) cropped_image.ptr(j, i)) + c * type_bytes_count;
				auto dst = ((unsigned char*) cropped_image_channel.ptr(j, i));
				for (size_t b = 0; b < type_bytes_count; ++b) {
					dst[b] = src[b];
				}
			}
		}
		cropped_image = cropped_image_channel;
	}

	return cropped_image;
}

size_t shared_device_image::width() const
{
	return width_;
}

size_t shared_device_image::height() const
{
	return height_;
}

size_t shared_device_image::cn() const
{
	return cn_;
}

DataType shared_device_image::dataType() const
{
	return data_type_;
}

bool shared_device_image::isNull() const
{
	return data_ == NULL;
}

void shared_device_image::reset()
{
	decref();
}

void shared_device_image::resize(size_t width, size_t height, DataType data_type)
{
	resize(width, height, 1, data_type);
}

void shared_device_image::resize(size_t width, size_t height, size_t cn, DataType data_type)
{
	rassert(cn >= 1, 332061633);
	if (width == width_ && height == height_ && cn == cn_ && data_type == data_type_)
		return;

	decref();

	Context context;
	Context::Type type = context.type();

	switch (type) {
#ifdef CUDA_SUPPORT
	case Context::TypeCUDA:
	{
		int xyzw_nbits[4] = {0, 0, 0, 0};
		cudaChannelFormatKind format;
		{
			for (int c = 0; c < cn; ++c) {
				xyzw_nbits[c] = 8 * dataSize(data_type);
			}
			if        (data_type == DataType8u || data_type == DataType16u || data_type == DataType32u) {
				format = cudaChannelFormatKindUnsigned;
			} else if (data_type == DataType8i || data_type == DataType16i || data_type == DataType32i) {
				format = cudaChannelFormatKindSigned;
			} else if (data_type == DataType32f) {
				// DataType16 is not supported because it makes difficult uniform type of sampled value from kernel 
				// In such a way it is always a float32: 
				//  - a nature float32 due to cudaReadModeElementType
				//  - a normalized float32 due to cudaReadModeNormalizedFloat (in case of integer data types)
				format = cudaChannelFormatKindFloat;
			} else {
				throw std::runtime_error("Unsupported data type: " + typeName(data_type));
			}
		}

		cudaArray_t cuArray = NULL;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(xyzw_nbits[0], xyzw_nbits[1], xyzw_nbits[2], xyzw_nbits[3], format);
		CUDA_SAFE_CALL(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore));

		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModeLinear;
		if (format == cudaChannelFormatKindFloat) {
			texDesc.readMode = cudaReadModeElementType; // cudaReadModeNormalizedFloat can't be used for floating point data
		} else {
			texDesc.readMode = cudaReadModeNormalizedFloat; // note that linear filtering mode is incompatible with cudaReadModeElementType for unsigned data types
		}
		texDesc.normalizedCoords = 0;

		cudaTextureObject_t cuTexture = 0;
		cudaError_t create_cu_texture_code = cudaCreateTextureObject(&cuTexture, &resDesc, &texDesc, NULL);
		if (create_cu_texture_code != cudaSuccess) {
			CUDA_SAFE_CALL(cudaFreeArray(cuArray));
			CUDA_SAFE_CALL(create_cu_texture_code);
		}

		cudaSurfaceObject_t cuSurface = 0;
		cudaError_t create_cu_surface_code = cudaCreateSurfaceObject(&cuSurface, &resDesc);
		if (create_cu_surface_code != cudaSuccess) {
			CUDA_SAFE_CALL(cudaDestroyTextureObject(cuTexture));
			CUDA_SAFE_CALL(cudaFreeArray(cuArray));
			CUDA_SAFE_CALL(create_cu_surface_code);
		}

		data_ = cuArray;
		cu_texture_ = cuTexture;
		cu_surface_ = cuSurface;
		break;
	}
#endif
	case Context::TypeOpenCL:
	{
		cl_mem image = context.cl()->createImage2D(width, height, cn, data_type);

		data_ = image;
		break;
	}
	case Context::TypeVulkan:
	{
		data_ = allocateVkImage(width, height, cn, data_type);
		vkImageData()->transitionLayout(vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
		break;
	}
	default:
		gpu::raiseException(_SHORT_FILE_, __LINE__, "No GPU context");
	}

	buffer_	= new unsigned char [8];
	* (long long *) buffer_ = 0;
	incref();

	type_		= type;

	width_		= width;
	height_		= height;
	cn_			= cn;
	data_type_	= data_type;
}

avk2::raii::ImageData* shared_device_image::allocateVkImage(unsigned int width, unsigned int height, size_t cn, DataType data_type)
{
	Context context;
	rassert(context.type() == Context::TypeVulkan, 291944745);
	return context.vk()->createImage2DArray(width, height, cn, data_type);
}

avk2::raii::ImageData* shared_device_depth_image::allocateVkImage(unsigned int width, unsigned int height, size_t cn, DataType data_type)
{
	Context context;
	rassert(context.type() == Context::TypeVulkan, 291944745);
	rassert(cn == 1, 892261498); // TODO check that we use Image2D instead of Image2DArray (with single layer)
	rassert(data_type == DataType32f, 926666669);
	return context.vk()->createDepthImage(width, height); // it differs mostly with different vk::Format - eD32Sfloat instead of eR32Sfloat - this makes possible to use image as depth framebuffer attachment
}

size_t shared_device_image::size() const
{
	return width_* height_ * cn_ * dataSize(data_type_);
}

}
