#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define RASSERT_ENABLED         1 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#define WG_SIZE                 128 // work group size
#define GROUP_SIZE              WG_SIZE
#define ELEMENTS_PER_WORK_ITEM  8
#define ELEMENTS_PER_CHUNK      (WG_SIZE * ELEMENTS_PER_WORK_ITEM) // elements per chunk/work group

#define BITS_PER_PASS           4
#define CASES_PER_PASS          (1 << BITS_PER_PASS) // how many different cases (combinations) of sorted bits we have
#define BITS_PER_PASS_MASK      ((1 << BITS_PER_PASS) - 1)

#define UINT32_INFINITY         0xffffffffu

#endif // pragma once
