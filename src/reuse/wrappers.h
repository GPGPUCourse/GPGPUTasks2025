#pragma once

#include <libgpu/vulkan/engine.h>

namespace gpuptr {
class u32 {
   public:
    explicit u32(gpu::gpu_mem_32u& origin) : buffer_(origin), base_(0), size_(origin.number()) {}
    explicit u32(gpu::gpu_mem_32u& origin, unsigned int base) : buffer_(origin), base_(base), size_(origin.number()) {}

    u32 allocate(unsigned int size) {
        rassert(size_ >= size, 7652847);
        auto res = *this;
        res.size_ = size;
        base_ += size;
        size_ -= size;
        return res;
    }

    int size() const { return size_; }
    auto cuptr() { return buffer_.cuptr() + base_; }
    auto cuptr() const { return buffer_.cuptr() + base_; }
    uint32_t at(int i) {
        rassert(i < size_, 3984765908);
        uint32_t res;
        buffer_.readN(&res, 1, base_ + i);
        return res;
    }

   private:
    gpu::gpu_mem_32u& buffer_;
    unsigned int base_;
    unsigned int size_;
};
}  // namespace gpuptr