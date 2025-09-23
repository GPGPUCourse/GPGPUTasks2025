#include "omp_utils.h"


Mutex exception_dispatcher::mutex_;

void exception_dispatcher::init(const std::exception_ptr &eptr){
	Lock lock(mutex_);

	if (eptr_ == std::exception_ptr())
		eptr_ = eptr;
}
