#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#ifdef __cplusplus
inline constexpr unsigned int BLOCK_THREADS = 256;
inline constexpr unsigned int THREAD_ELEMS = 32;
inline constexpr unsigned int BLOCK_ELEMS = BLOCK_THREADS * THREAD_ELEMS;
inline constexpr unsigned int WARPS_CNT = BLOCK_THREADS >> 5;
inline constexpr unsigned int BITS_AT_A_TIME = 4;
inline constexpr unsigned int BINS_CNT = 1u << BITS_AT_A_TIME;
inline constexpr unsigned int WARP_BINS_CNT = WARPS_CNT << BITS_AT_A_TIME;
inline constexpr unsigned int BINS_IN_NUM = sizeof(unsigned int) << 1; // (sizeof(unsigned int) * <bits in byte>) / BITS_AT_A_TIME
#endif

#endif // pragma once
