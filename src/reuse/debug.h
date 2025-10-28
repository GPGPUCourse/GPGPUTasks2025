#pragma once

#include "kernels/kernels.h"

#include <fstream>
#include <iomanip>
#include <iostream>

#define DEBUG false
// #define DEBUG true

void print(std::string s)
{
    if (!DEBUG) {
        return;
    }
    std::cout << s;
}

void printVec(std::string label, gpu::gpu_mem_32u& a, int base, int sz, std::string post)
{
    if (!DEBUG) {
        return;
    }
    std::cout << label;
    std::cout << '[' << a.cuptr() + base << ", " << a.cuptr() + base + sz << ')';
    std::cout << ": ";
    auto tmp = a.readVector();
    for (int i = 0; i < sz; ++i) {
        if (base + i == tmp.size()) {
            std::cout << "OUT OF BOUND";
            break;
        }
        std::cout << std::setw(4) << tmp[base + i] << ' ';
    }
    if (base + sz == tmp.size()) {
        std::cout << "[OUT OF BOUND]";
    } else {
        std::cout << "[" << tmp[base + sz] << "]";
    }
    std::cout << post;
}