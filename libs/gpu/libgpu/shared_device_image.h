#pragma once

#include "shared_device_buffer.h"
#include <libbase/data_type.h>
#include <libimages/images.h>

#include <cstddef>


typedef unsigned long long cudaSurfaceObject_t;
typedef unsigned long long cudaTextureObject_t;
typedef struct _cl_mem *cl_mem;

namespace avk2 {
	namespace raii {
		class ImageData;
	}
}

namespace gpu {

class shared_device_image {
public:
	shared_device_image();
	shared_device_image(size_t width, size_t height, DataType data_type)			{ resize(width, height, data_type); }
	shared_device_image(size_t width, size_t height, size_t cn, DataType data_type)	{ resize(width, height, cn, data_type); }
	~shared_device_image();
	shared_device_image(const shared_device_image &other);
	shared_device_image &operator= (const shared_device_image &other);

	void			swap(shared_device_image &other);
	void			reset();

	size_t			width() const;
	size_t			height() const;
	size_t			cn() const;
	DataType		dataType() const;

	void			resize(size_t width, size_t height, DataType data_type);
	void			resize(size_t width, size_t height, size_t cn, DataType data_type);
	size_t			size() const;
	bool 			isNull() const;

	cudaSurfaceObject_t	cuSurface() const; // for write-only
	cudaTextureObject_t	cuTexture() const; // for read-only

	cl_mem				clmem() const;

	avk2::raii::ImageData	*vkImageData() const;

	void				write(const shared_device_buffer &buffer, size_t width, size_t height, size_t src_offset, size_t dst_x_offset, size_t dst_y_offset, bool async);
	void				write(const AnyImage &image);
	AnyImage			read() const;
	AnyImage			read(size_t width, size_t height, size_t offset_x=0, size_t offset_y=0, int c=-1) const;

	static shared_device_image create(size_t width, size_t height, size_t cn, DataType data_type);

protected:
	virtual avk2::raii::ImageData* allocateVkImage(unsigned int width, unsigned int height, size_t cn, DataType data_type);

	void	incref();
	void	decref();

	unsigned char *		buffer_;
	void *				data_;
	cudaTextureObject_t	cu_texture_;
	cudaSurfaceObject_t	cu_surface_;
	int					type_;

	size_t				width_;
	size_t				height_;
	size_t				cn_;
	DataType			data_type_;
};

template <typename T>
class shared_device_image_typed : public shared_device_image {
public:
	shared_device_image_typed()											{}
	shared_device_image_typed(size_t width, size_t height)				{ resize(width, height); }
	shared_device_image_typed(size_t width, size_t height, size_t cn)	{ resize(width, height, cn); }

	void			resize(size_t width, size_t height)					{ shared_device_image::resize(width, height, DataTypeTraits<T>::type()); }
	void			resize(size_t width, size_t height, size_t cn)		{ shared_device_image::resize(width, height, cn, DataTypeTraits<T>::type()); }

	TypedImage<T> read() const																				{ return TypedImage<T>(shared_device_image::read()); }
	TypedImage<T> read(size_t width, size_t height, size_t offset_x=0, size_t offset_y=0, int c=-1) const	{ return TypedImage<T>(shared_device_image::read(width, height, offset_x, offset_y, c)); }

	void			fill(T value) {
		// TODO speedup
		TypedImage<T> image(width_, height_, cn_);
		image.fill(value);
		write(image);
	}
};

class shared_device_depth_image : public shared_device_image_typed<float> {
public:
	shared_device_depth_image()								{}
	shared_device_depth_image(size_t width, size_t height)	{ resize(width, height); }

protected:
	virtual avk2::raii::ImageData* allocateVkImage(unsigned int width, unsigned int height, size_t cn, DataType data_type) override;
};

}

typedef gpu::shared_device_image_typed<int8_t>		gpu_image8i;
typedef gpu::shared_device_image_typed<int16_t>		gpu_image16i;
typedef gpu::shared_device_image_typed<int32_t>		gpu_image32i;
typedef gpu::shared_device_image_typed<uint8_t>		gpu_image8u;
typedef gpu::shared_device_image_typed<uint16_t>	gpu_image16u;
typedef gpu::shared_device_image_typed<uint32_t>	gpu_image32u;
typedef gpu::shared_device_image_typed<float>		gpu_image32f;

typedef gpu::shared_device_depth_image				gpu_imageDepth;

