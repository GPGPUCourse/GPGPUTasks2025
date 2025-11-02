#pragma once

#include "kernels.h"
#include "wrappers.h"

void prefixSum(gpuptr::u32 in, gpuptr::u32 out, gpuptr::u32 buffer);