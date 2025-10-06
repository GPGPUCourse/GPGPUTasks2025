#include "omp_utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif


int getOpenMPThreadsCount()
{
    int parallel_threads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp master
        parallel_threads = omp_get_num_threads(); // threads actually spawned
    }
#endif
    return parallel_threads;
}

Mutex exception_dispatcher::mutex_;

void exception_dispatcher::init(const std::exception_ptr &eptr){
	Lock lock(mutex_);

	if (eptr_ == std::exception_ptr())
		eptr_ = eptr;
}
