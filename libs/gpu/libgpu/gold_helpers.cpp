#include "gold_helpers.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace gold {

    template<typename T>
    void host_data<T>::init(const gpu::gpu_mem_any &gpu_data) {
        size_t n = gpu_data.size() / sizeof(T);
        data = std::vector<T>(n);
        gpu_data.read(data.data(), gpu_data.size());
    }

    template<typename T>
    void host_data<T>::init(const gpu::shared_device_buffer_typed <T> &gpu_data) {
        size_t n = gpu_data.size() / sizeof(T);
        data = std::vector<T>(n);
        gpu_data.readN(data.data(), n);
    }

    template<typename T>
    bool host_data<T>::operator==(const host_data <T> &that) {
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] != that.data[i]) {
                return false;
            }
        }
        return true;
    }

    template<typename T>
    T diff(T a, T b) {
        return std::max(a, b) - std::min(a, b);
    }

    float diff(float a, float b) {
        if (!std::isnan(a) && std::isnan(b)) {
            return std::max(a, b) - std::min(a, b);
        } else if (std::isnan(a) &&std::isnan(b)) {
            return 0.0f;
        } else if (std::isnan(a)) {
            return std::abs(b);
        } else {
            assert(std::isnan(b));
            return std::abs(a);
        }
    }

    double diff(double a, double b) {
        if (std::isnan(a) && std::isnan(b)) {
            return std::max(a, b) - std::min(a, b);
        } else if (std::isnan(a) &&std::isnan(b)) {
            return 0.0;
        } else if (std::isnan(a)) {
            return std::abs(b);
        } else {
            assert(std::isnan(b));
            return std::abs(a);
        }
    }

    void ensure(bool condition, int line) {
        if (!condition) {
            std::cerr << "GOLD check filed at line " << line << std::endl;
        }
    }

    template <typename T>
    void ensure_less(T a, T b, int line) {
        if (a < b) {
            return;
        } else {
            std::cerr << "Failed check: " << a << " < " << b << std::endl;
            ensure(a < b, line);
        }
    }

    template class host_data<int8_t>;
    template class host_data<int16_t>;
    template class host_data<int32_t>;
    template class host_data<uint8_t>;
    template class host_data<uint16_t>;
    template class host_data<uint32_t>;
    template class host_data<float>;
    template class host_data<double>;

    template void ensure_less(uint8_t a, uint8_t b, int line);
    template void ensure_less(uint16_t a, uint16_t b, int line);
    template void ensure_less(uint32_t a, uint32_t b, int line);
    template void ensure_less(float a, float b, int line);
    template void ensure_less(double a, double b, int line);

}
