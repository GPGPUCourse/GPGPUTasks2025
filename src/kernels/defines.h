#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
// #define GROUP_SIZE   128 // - для запуска на CPU, иначе было Error: Too big workgroup size for this kernel: 256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#endif // pragma once
