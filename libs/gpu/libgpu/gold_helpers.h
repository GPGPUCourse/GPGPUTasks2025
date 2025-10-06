#pragma once

#include "shared_device_buffer.h"

#include <vector>

#define GOLD_CHECK(condition) gold::ensure(condition, __LINE__)
#define GOLD_CHECK_LESS(a, b) gold::ensure_less(a, b, __LINE__)

namespace gold {

	template <typename T>
	class host_data {
	public:
		host_data() {}
		host_data(const gpu::gpu_mem_any& gpu_data)						{ init(gpu_data); };
		host_data(const gpu::shared_device_buffer_typed<T>& gpu_data)	{ init(gpu_data); };

		void init(const gpu::gpu_mem_any& gpu_data);
		void init(const gpu::shared_device_buffer_typed<T>& gpu_data);

		bool operator==(const host_data<T>& that);
		bool operator!=(const host_data<T>& that) { return !(*this == that); }

		T* ptr() { return data.data(); }

	private:
		std::vector<T> data;
	};

	void ensure(bool condition, int line);

	template <typename T>
	void ensure_less(T a, T b, int line);

}
