#include "../../defines.h"

#if RASSERT_ENABLED
    #define rassert(condition, error_code)					                    \
            do {									            \
                    if (!(condition)) {						                    \
                            printf("rassert code=%d line=%d\n", error_code % 1000000000, __LINE__); \
                    }								                    \
            } while (false)
#else
    #define rassert(condition, error_code) // do nothing
#endif
