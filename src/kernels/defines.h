#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

// Radix-sort tuning knobs (shared between host and device code)
#define RADIX_BITS        4u
#define RADIX_BUCKETS     (1u << RADIX_BITS)
#define ITEMS_PER_THREAD  4u

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
