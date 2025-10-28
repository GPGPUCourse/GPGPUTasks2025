#pragma once

#include <cstdarg>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "kernels.h"
#include "wrappers.h"

#define DEBUG false
// #define DEBUG true

inline int print(const char* fmt, ...) {
    if (!DEBUG) {
        return 0 ;
    }
    va_list ap;
    va_start(ap, fmt);
    va_list ap_copy;
    va_copy(ap_copy, ap);

    int len = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);

    if (len < 0) {
        va_end(ap_copy);
        return len;  // ошибка форматирования
    }

    std::vector<char> buf(static_cast<size_t>(len) + 1);
    std::vsnprintf(buf.data(), buf.size(), fmt, ap_copy);
    va_end(ap_copy);

    std::cout.write(buf.data(), len);
    return len;  // как printf: количество выведенных символов (без завершающего '\0')
}


inline void printVec(std::string label, gpuptr::u32 a, int sz, std::string post) {
    if (!DEBUG) {
        return;
    }
    std::cout << label;
    std::cout << '[' << a.cuptr() << ", " << a.cuptr() + sz << ')';
    std::cout << ": ";
    for (int i = 0; i < sz; ++i) {
        if (i == a.size()) {
            std::cout << "OUT OF BOUND";
            break;
        }
        std::cout << std::setw(4) << a.at(i) << ' ';
    }

    if (sz == a.size()) {
        std::cout << "[OUT OF BOUND]";
    } else {
        std::cout << "[" << a.at(sz) << "]";
    }
    std::cout << post;
}

inline std::string toBin(uint32_t x) {
    std::string s(32, '0');
    for (int i = 0; i < 32; ++i) {
        s[32 - 1 - i] = '0' + x % 2;
        x /= 2;
    }
    return s;
}
inline void printVecBin(std::string label, gpuptr::u32& a, int sz, std::string post) {
    if (!DEBUG) {
        return;
    }
    std::cout << label;
    std::cout << '[' << a.cuptr() << ", " << a.cuptr() + sz << ')';
    std::cout << ":" << '\n';
    for (int i = 0; i < sz; ++i) {
        if (i == a.size()) {
            std::cout << "OUT OF BOUND";
            break;
        }

        std::cout << i << ' ' << toBin(a.at(i)) << '\n';
    }

    if (sz == a.size()) {
        std::cout << "[OUT OF BOUND]";
    } else {
        std::cout << "[" << a.at(sz) << "]";
    }
    std::cout << post;
}