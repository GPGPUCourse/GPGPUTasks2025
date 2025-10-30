#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define BITS_PER_PASS 4
#define NUM_DIGITS (1 << BITS_PER_PASS)
#define RADIX_MASK (NUM_DIGITS - 1)

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
