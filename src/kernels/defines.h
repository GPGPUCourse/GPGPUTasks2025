#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE_POW   8
#define GROUP_SIZE   (1 << 8)
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define AO_SAMPLES   8

#define BOX_BLOCK_SIZE 8

#define BIG_BLOCK_POW 4
#define BIG_BLOCK_SIZE (1 << BIG_BLOCK_POW)

#define SMALL_BLOCK_POW 1//3
#define SMALL_BLOCK_SIZE (1 << SMALL_BLOCK_POW)

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
