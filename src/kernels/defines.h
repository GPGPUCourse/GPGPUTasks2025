#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE_LOG 8
#define GROUP_SIZE 256

#define BIT_GRANULARITY 6 // amount of bits processed at a time
#define BIT_GRANULARITY_EXP (1 << BIT_GRANULARITY)
#define GRANULARITY_MASK (BIT_GRANULARITY_EXP - 1)

#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
