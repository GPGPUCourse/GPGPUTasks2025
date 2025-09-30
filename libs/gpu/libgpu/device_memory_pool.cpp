#include "device_memory_pool.h"
#include <algorithm>
#include <sstream>
#include <vector>

namespace gpu {

shared_device_buffer device_memory_pool::allocate(const std::string &name, size_t size)
{
	std::map<std::string, shared_device_buffer>::const_iterator it = buffers_.find(name);
	if (it != buffers_.end()) {
		if (it->second.size() >= size)
			return it->second;
		buffers_.erase(it);

		// overallocate 10% more to reduce frequent realocation
		size += size / 10;
	}

	shared_device_buffer res = shared_device_buffer::create(size);
	buffers_[name] = res;
	return res;
}

void device_memory_pool::release(const std::string &name)
{
	buffers_.erase(name);
}

shared_device_buffer device_memory_pool::get(const std::string &name)
{
	return allocate(name, 0);
}

void device_memory_pool::clear()
{
	buffers_.clear();
}

size_t device_memory_pool::size(const std::string &group) const
{
	size_t res = 0;
	for (std::map<std::string, shared_device_buffer>::const_iterator it = buffers_.begin(); it != buffers_.end(); ++it) {
		const std::string &name = it->first;
		if (group.size() && name.substr(0, group.size()) != group)
			continue;
		res += it->second.size();
	}
	return res;
}

std::string device_memory_pool::info() const
{
	std::vector< std::pair<size_t, std::string> > buffers;

	size_t total_size = 0;
	for (std::map<std::string, shared_device_buffer>::const_iterator it = buffers_.begin(); it != buffers_.end(); ++it) {
		size_t size = it->second.size();
		total_size += size;
		buffers.push_back(std::make_pair(size, it->first));
	}

	std::sort(buffers.rbegin(), buffers.rend());

	std::ostringstream out;
	out << "allocated " << buffers.size() << " buffers, " << (total_size >> 20) << " MB" << std::endl;

	bool truncated = false;
	if (buffers.size() > 10) {
		buffers.resize(10);
		truncated = true;
	}

	for (size_t i = 0; i < buffers.size(); i++) {
		const std::string &name = buffers[i].second;
		size_t size = buffers[i].first;
		out << "  " << name << ": " << (size >> 20) << " MB" << std::endl;
	}

	if (truncated)
		out << "  ..." << std::endl;

	return out.str();
}

}
