#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define UINT_MAX 4294967295    // 2^32 − 1

inline void minimax(__local uint* a, __local uint* b, int rev) {
    uint tmp;
    if ((long long)(*a) * rev > (long long)(*b) * rev) {
        tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

inline void giant_stepper(int giant_step, uint thr, __local uint* cache) {
    for (int baby_step = giant_step / 2; baby_step > 1; baby_step /= 2) {
        if (thr % baby_step < baby_step / 2) {
            int rev = (thr % giant_step >= giant_step / 2 ? -1 : 1);
            minimax(&cache[thr], &cache[thr + baby_step / 2], rev);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void bitonic_sort_256(
    __global unsigned int* a,
    int  n)
{
    unsigned int i = get_global_id(0);
    unsigned int thr = get_local_id(0);

    __local uint cache[2 * GROUP_SIZE];
    if (i < n) {
        cache[thr] = a[i];
        cache[thr + GROUP_SIZE] = UINT_MAX;
    }
    else {
        cache[thr] = UINT_MAX;
        cache[thr + GROUP_SIZE] = UINT_MAX;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    giant_stepper(4, thr, cache);
    giant_stepper(8, thr, cache);
    giant_stepper(16, thr, cache);
    giant_stepper(32, thr, cache);
    giant_stepper(64, thr, cache);
    giant_stepper(128, thr, cache);
    giant_stepper(256, thr, cache);
    giant_stepper(512, thr, cache);
    
    
    // if (thr % 4 < 2) {
    //     int rev = (thr % 8 >= 4 ? -1 : 1);
    //     minimax(&cache[thr], &cache[thr + 2], rev);
    // }
    // if (thr % 2 < 1) {
    //     int rev = (thr % 8 >= 4 ? -1 : 1);
    //     minimax(&cache[thr], &cache[thr + 1], rev);
    // }
    
    
    


    // if (thr % 8 < 4)
    //     minimax(&cache[thr], &cache[thr + 4]);
    // if (thr % 4 < 2)
    //     minimax(&cache[thr], &cache[thr + 2]);
    // if (thr % 2 < 0)
    //     minimax(&cache[thr], &cache[thr + 1]);


    if (i < n)
        a[i] = cache[thr];
}
