#pragma once

#include "shared_device_buffer.h"
#include <string>
#include <map>

namespace gpu {

class device_memory_pool {
public:
	shared_device_buffer	allocate(const std::string &name, size_t size);
	void					release(const std::string &name);
	shared_device_buffer	get(const std::string &name);

	void					clear();
	size_t					size(const std::string &group = std::string()) const;
	std::string				info() const;

	template <typename T>
	class shared_device_buffer_typed<T> allocateN(const std::string &name, size_t count)
	{
		return shared_device_buffer_typed<T>(allocate(name, count * sizeof(T)));
	}

protected:
	std::map<std::string, shared_device_buffer>	buffers_;
};

}
