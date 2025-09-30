#pragma once

#include <cstddef>
#include <stdint.h>

namespace gpu {

class shared_host_buffer {
public:
	shared_host_buffer();
	~shared_host_buffer();
	shared_host_buffer(const shared_host_buffer &other);
	shared_host_buffer &operator= (const shared_host_buffer &other);

	void			swap(shared_host_buffer &other);
	void *			get() const;
	size_t			size() const;
	void			resize(size_t size);
	void			grow(size_t size);

	static shared_host_buffer create(size_t size);

protected:
	void	incref();
	void	decref();

	unsigned char *	buffer_;
	void *			data_;
	int				type_;
	size_t			size_;
};

template <typename T>
class shared_host_buffer_typed : public shared_host_buffer {
public:
	void			resizeN(size_t number);

	T *				get() const;

	size_t			number() const;

	static shared_host_buffer_typed<T> createN(size_t number);
};

typedef shared_host_buffer							gpu_host_mem_any;

typedef shared_host_buffer_typed<int16_t>			gpu_host_mem_16i;
typedef shared_host_buffer_typed<int32_t>			gpu_host_mem_32i;
typedef shared_host_buffer_typed<uint8_t>			gpu_host_mem_8u;
typedef shared_host_buffer_typed<uint16_t>			gpu_host_mem_16u;
typedef shared_host_buffer_typed<uint32_t>			gpu_host_mem_32u;
typedef shared_host_buffer_typed<float>				gpu_host_mem_32f;

}
