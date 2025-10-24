#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16
#define DIGITS_PER_BLOCK 4
#define BUCKETS (1 << DIGITS_PER_BLOCK)
#define COUNTING_MASK (BUCKETS - 1)
#define WORKITEM_MASK ((GROUP_SIZE >> DIGITS_PER_BLOCK) - 1)
#define WORKITEM_MASK_BIT (8 - DIGITS_PER_BLOCK)

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
