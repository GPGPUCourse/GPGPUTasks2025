#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

//for radix sort
#define BITS_PER_PASS 2
#define RADIX (1u << BITS_PER_PASS)
#define MASK (RADIX - 1)

#endif // pragma once
