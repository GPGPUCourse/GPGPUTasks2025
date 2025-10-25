#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define BUCKET_BITS  4
#define BUCKET_COUNT (1 << BUCKET_BITS)
#define BUCKET_MASK (BUCKET_COUNT - 1)
#define ELEMENT_SIZE 32

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
