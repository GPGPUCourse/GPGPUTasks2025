#pragma once

#include "libgpu/shared_device_buffer.h"

#define uint unsigned int

namespace models {

namespace gpu {
struct CSRMatrix;
}

inline void print(const ::gpu::gpu_mem_32u& array)
{
    std::vector<uint> vec = array.readVector();
    for (auto x : vec) std::cout << x << " ";
    std::cout << std::endl;
}

struct CSRMatrix {
    ::gpu::gpu_mem_32u values;
    ::gpu::gpu_mem_32u cols;
    ::gpu::gpu_mem_32u offsets;

    CSRMatrix(::gpu::gpu_mem_32u values, ::gpu::gpu_mem_32u cols, ::gpu::gpu_mem_32u offsets) : values(std::move(values)), cols(std::move(cols)), offsets(std::move(offsets)) {}

    [[nodiscard]] gpu::CSRMatrix to_gpu() const;

    void print() const
    {
        std::cout << " values: ";
        ::models::print(values);
        std::cout << "   cols: ";
        ::models::print(cols);
        std::cout << "offsets: ";
        ::models::print(offsets);
    }
};

namespace gpu {

struct Array {
    const uint* array;
    size_t length;

    explicit Array(const ::gpu::gpu_mem_32u& array) : array(array.cuptr()), length(array.number()) {}
};

struct CSRMatrix {
    Array values;
    Array cols;
    Array offsets;

    explicit CSRMatrix(const ::models::CSRMatrix& matrix) : values(matrix.values), cols(matrix.cols), offsets(matrix.offsets) {}
};

}

inline gpu::CSRMatrix CSRMatrix::to_gpu() const
{
    return gpu::CSRMatrix(*this);
}

}
