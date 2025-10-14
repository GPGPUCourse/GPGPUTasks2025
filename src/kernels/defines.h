#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 32
#define GROUP_SIZE_Y 32

#define VEC_SIZE_V 7
#define VEC_SIZE_H 7
#define TILE GROUP_SIZE_X

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
