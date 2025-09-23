#pragma once

#define T_DEPENDENT2(fun, suffix) fun ## _ ## suffix
#define T_DEPENDENT1(fun, suffix) T_DEPENDENT2(fun, suffix)
#define T_DEPENDENT(fun) T_DEPENDENT1(fun, T)
