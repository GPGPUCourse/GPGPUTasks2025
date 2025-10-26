#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#define BITS_IN_RADIX_SORT_ITERATION 4
#define NUM_BOXES 16
#define RADIX_MASK 15

#define WARP_SIZE 32
#define WARP_LG 5

#endif // pragma once
