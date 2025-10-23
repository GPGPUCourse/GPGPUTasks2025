#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16
#define LOG_GROUP_SIZE 8
#define RADIX 2
#define RADIX_MASK ((1 << RADIX) - 1)
#define BIN_COUNT (1 << RADIX)
#define MAX (~0)
#define COMPRESSED(n) ((n + GROUP_SIZE - 1) / GROUP_SIZE)

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
