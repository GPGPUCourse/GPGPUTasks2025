#pragma once

#if defined _WIN32 || defined _WIN64
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <pthread.h>
#endif

class Lock;

class Mutex {
public:
	Mutex ();
	virtual ~Mutex ();

	void	lock () const;
	void	unlock () const;
	bool	tryLock () const;

private:
#if defined _WIN32 || defined _WIN64
	mutable CRITICAL_SECTION _mutex;
#else
	mutable pthread_mutex_t _mutex;
#endif

	void operator = (const Mutex& M);
	Mutex (const Mutex& M);

	friend class Lock;
};

class Lock {
public:
	Lock (const Mutex& m, bool autoLock = true) : _mutex (m), _locked (false)
	{
		if (autoLock) {
			_mutex.lock();
			_locked = true;
		}
	}

	~Lock ()
	{
		if (_locked)
			_mutex.unlock();
	}

	void acquire ()
	{
		_mutex.lock();
		_locked = true;
	}

	void release ()
	{
		_mutex.unlock();
		_locked = false;
	}

	bool locked ()
	{
		return _locked;
	}

private:
	const Mutex &	_mutex;
	bool			_locked;
};

class TryLock {
public:
	TryLock (const Mutex& m, bool autoLock = true) : _mutex (m), _locked (false)
	{
		if (autoLock)
			_locked = _mutex.tryLock();
	}

	~TryLock ()
	{
		if (_locked)
			_mutex.unlock();
	}

	bool acquire ()
	{
		_locked = _mutex.tryLock();
	}

	void release ()
	{
		_mutex.unlock();
		_locked = false;
	}

	bool locked ()
	{
		return _locked;
	}

private:
	const Mutex &	_mutex;
	bool			_locked;
};

class MutexPool {
public:
	MutexPool(size_t size = 256);
	~MutexPool();

	Mutex &get(const void *address);

	static MutexPool *instance();

private:
	typedef Mutex *	MutexPtr;

	Mutex		mutex_;
	MutexPtr *	mutexes_;
	size_t		size_;
};
