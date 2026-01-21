#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define NUM_BOXES 256
#define RADIX_MASK 0xFF
#define BITS_IN_RADIX_SORT_ITERATION 8
#define WARP_LG 5

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
