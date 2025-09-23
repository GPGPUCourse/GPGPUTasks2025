#pragma once

#include "thread_mutex.h"

#include <stdexcept>


class exception_dispatcher {
public:
	void init(const std::exception_ptr &eptr);

	void init()
	{
		init(std::current_exception());
	}

	void reset()
	{
		eptr_ = std::exception_ptr();
	}

	bool loaded() const
	{
		return !(eptr_ == std::exception_ptr());
	}

	void dispatch() const
	{
		if (!(eptr_ == std::exception_ptr()))
			std::rethrow_exception(eptr_);
	}

protected:
	std::exception_ptr	eptr_;
	static Mutex		mutex_;
};

#define OMP_DISPATCHER_INIT exception_dispatcher __dispatcher__;
#define OMP_TRY {if (!__dispatcher__.loaded()) { try {
#define OMP_CATCH } catch (...) {	__dispatcher__.init();	}}}
#define OMP_RETHROW __dispatcher__.dispatch();
#define OMP_CATCH_RETHROW } catch (...) {	__dispatcher__.init();	}}}; __dispatcher__.dispatch();
