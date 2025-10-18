#pragma once

#include "runtime_assert.h"

#include <limits>


template <typename T>
T div_ceil(T num, T denom) {
    static_assert(std::numeric_limits<T>::is_integer, "T should be integer");
    rassert(num >= 0 && denom > 0, 237281390023106);
    rassert(num <= std::numeric_limits<T>::max() - (denom - 1), 23819230180012306); // overflow check
    return (num + denom - 1) / denom;
}

template<class TargetT, class SourceT>
TargetT narrow_cast(SourceT value)
{
	TargetT casted = value;

	bool is_different_sign = (value < (SourceT) 0) != (casted < (TargetT) 0);
	rassert((SourceT) casted == value && !is_different_sign, 452134143211);

	return casted;
}
