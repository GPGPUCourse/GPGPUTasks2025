#pragma once

#include <libgpu/vulkan/engine.h>

namespace gpuptr {
class u32 {
   public:
    explicit u32(gpu::gpu_mem_32u& origin) : buffer_(origin), base_(0) {}
    explicit u32(gpu::gpu_mem_32u& origin, unsigned int base)
        : buffer_(origin), base_(base) {}

    u32& operator+=(int x) {
        base_ += x;
        return *this;
    }

    friend u32 operator+(u32 a, int x) {
        a += x;
        return a;
    }

    int size() const { return buffer_.number() - base_; }

    auto cuptr() { return buffer_.cuptr() + base_; }

    auto cuptr() const { return buffer_.cuptr() + base_; }

   private:
    gpu::gpu_mem_32u& buffer_;
    unsigned int base_;
};
}  // namespace gpuptr