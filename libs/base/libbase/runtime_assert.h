#pragma once

#include <sstream>
#include <assert.h>

// working copy: true, trunk: false; this file can be added to ignore-on-commit changelist
// to enable/disable locally add   #define ENABLE_DEV_ASSERTS false   before includes in your .cpp
#ifndef ENABLE_DEV_ASSERTS
#define ENABLE_DEV_ASSERTS false
#endif

// breakpoints can be set here to debug assertion raises
[[maybe_unused]] static int debugPoint(int line)
{
	if (line < 0)
		return 0;
	return line;
}

inline void rassert_print(std::ostream &stream)
{
}

template <typename First, typename ...Rest>
inline void rassert_print(std::ostream &stream, const First &first, const Rest &...rest)
{
	stream << first;
	if (sizeof...(rest) > 0)
		stream << " ";
	rassert_print(stream, rest...);
}

template <typename Exception, typename ErrorCode, typename ...Args>
[[ noreturn ]] inline void rassert_throw(const ErrorCode &error_code, int line, const Args &...args)
{
	std::stringstream ss;
	ss << "Assertion \"";
	rassert_print(ss, error_code, args...);
	ss << "\" failed at line " << debugPoint(line);
	std::stringstream ec;
	ec << error_code;
	throw Exception(ss.str(), ec.str());
}

// disabled dassert (and do-while trick) http://web.archive.org/web/20201129200055/http://cnicholson.net/2009/02/stupid-c-tricks-adventures-in-assert/
// do-while trick: https://stackoverflow.com/questions/1067226/c-multi-line-macro-do-while0-vs-scope-block

// common macros
#define rassert_ex(condition, exception_type, error_code, ...)						\
	do {																			\
		if (!(condition)) {															\
			rassert_throw<exception_type>((error_code), __LINE__, ##__VA_ARGS__);	\
		}																			\
	} while (0)

class assertion_error : public std::runtime_error {
public:
    assertion_error(const std::string &message, const std::string &code = std::string()) : std::runtime_error(message), code_(code) { }
    assertion_error(const char *message, const std::string &code = std::string()) : std::runtime_error(message), code_(code) { }

    const std::string &code() const
    {
        return code_;
    }

protected:
    std::string code_;
};

class gpu_failure : public assertion_error {
public:
    gpu_failure(const std::string &message, const std::string &code = std::string()) : assertion_error(message, code) { }
    gpu_failure(const char *message, const std::string &code = std::string()) : assertion_error(message, code) { }
};

#define rassert(condition, error_code, ...)									\
	rassert_ex((condition), assertion_error, (error_code), ##__VA_ARGS__)

// disabled dassert (and do-while trick) http://web.archive.org/web/20201129200055/http://cnicholson.net/2009/02/stupid-c-tricks-adventures-in-assert/
// do-while trick: https://stackoverflow.com/questions/1067226/c-multi-line-macro-do-while0-vs-scope-block
#if ENABLE_DEV_ASSERTS
#define dassert(condition, error_code, ...)									\
	rassert((condition), (error_code), ##__VA_ARGS__)
#else
#define dassert(condition, error_code, ...)									\
	do {																	\
		(void) sizeof(condition);											\
	} while (0)
#endif

// call only for invariants that highly likely can only be violated by corrupted memory
#define rassert_memory(condition, error_code, ...)							\
	rassert_ex((condition), memory_failure, (error_code), ##__VA_ARGS__)

// call only for invariants that highly likely can only be violated by faulty gpu
#define rassert_gpu(condition, error_code, ...)								\
	rassert_ex((condition), gpu_failure, (error_code), ##__VA_ARGS__)

static inline void fassert(const int line, bool condition, const char *message)
{
	if (!condition) {
		std::stringstream ss;
		ss << "Assertion " << message << " failed at line " << line << "!";
		throw std::runtime_error(ss.str());
	}
	assert(condition);
}

#define frassert(condition, message) (fassert(__LINE__, (condition), (message)))

// usage:
// rassert(condition, id); //-- as usual
// rassert(p.x() >= 0, "Wrong position", p.x(), p.y()); //-- Assertion "Wrong position -5 9" failed at line 26!
// dassert(condition, id); //-- as usual, but enabled globally by ENABLE_DEV_ASSERTS in working copy, can be disabled in .cpp locally

// also can be defined (as in gtest)
//#define rassert_eq(val_a, val_b, message, ...) rassert(((val_a) == (val_b)), (message), (val_a), (val_b), ##__VA_ARGS__)
