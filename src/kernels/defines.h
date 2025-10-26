#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256u
#define GROUP_SIZE_X 16u
#define GROUP_SIZE_Y 16u
#define RADIX_BITS        4u
#define RADIX_BINS        (1u << (RADIX_BITS))    
#define BINS_IN_NUM       (32u / (RADIX_BITS))    
#define WARPS_PER_BLOCK   (((GROUP_SIZE) + 31u) / 32u)
#define WARP_BINS_CNT     ((WARPS_PER_BLOCK) * (RADIX_BINS))
#define ITEMS_PER_THREAD  8u
#define ITEMS_PER_BLOCK   ((GROUP_SIZE) * (ITEMS_PER_THREAD))


#endif // pragma once
