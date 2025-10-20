#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define ELEM_PER_THREAD 4
#define VEC_SIZE        4
#define GROUP_SIZE      256
#define CHUNK_SIZE      (GROUP_SIZE * ELEM_PER_THREAD)

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
