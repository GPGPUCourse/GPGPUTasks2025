#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#define BUCKET_BIT_SIZE 4
#define BUCKET_SIZE (1 << BUCKET_BIT_SIZE)
#define BUCKET_MASK (BUCKET_SIZE - 1)


#endif // pragma once
