#include "thread_mutex.h"
#include <cassert>

#define MUTEX_POOL_CHECK_FOR_DEADLOCKS 0

#if defined _WIN32 || defined _WIN64

// Windows threads

Mutex::Mutex()
{
	::InitializeCriticalSection(&_mutex);
}

Mutex::~Mutex()
{
	::DeleteCriticalSection(&_mutex);
}

void Mutex::lock() const
{
	::EnterCriticalSection(&_mutex);
	assert(_mutex.RecursionCount == 1);
}

void Mutex::unlock() const
{
	assert(_mutex.RecursionCount == 1);
	::LeaveCriticalSection(&_mutex);
}

bool Mutex::tryLock () const
{
	return (::TryEnterCriticalSection(&_mutex) != 0);
}

#else

// Posix threads

#include <errno.h>

Mutex::Mutex ()
{
	int error = ::pthread_mutex_init(&_mutex, 0);
	assert(error == 0);
}

Mutex::~Mutex ()
{
	int error = ::pthread_mutex_destroy(&_mutex);
	assert(error == 0);
}

void Mutex::lock() const
{
	int error = ::pthread_mutex_lock(&_mutex);
	assert(error == 0);
}

void Mutex::unlock() const
{
	int error = ::pthread_mutex_unlock(&_mutex);
	assert(error == 0);
}

bool Mutex::tryLock () const
{
	int error = ::pthread_mutex_trylock(&_mutex);
	if (error == EBUSY) return false;
	assert(error == 0);
	return true;
}

#endif

MutexPool global_mutexpool;

MutexPool::MutexPool(size_t size)
{
	size_		= size;
	mutexes_	= new MutexPtr[size];
	for (size_t k = 0; k < size; k++)
		mutexes_[k] = 0;
}

MutexPool::~MutexPool()
{
	for (size_t k = 0; k < size_; k++) {
		delete mutexes_[k];
		mutexes_[k] = 0;
	}
	delete[] mutexes_;
}

MutexPool *MutexPool::instance()
{
	return &global_mutexpool;
}

Mutex &MutexPool::get(const void *address)
{
	Lock lock(mutex_);

	size_t index = int(((size_t)(void *)(address) >> (sizeof(address) >> 1)) % size_);

#if MUTEX_POOL_CHECK_FOR_DEADLOCKS
	index = 0;
#endif

	Mutex *m = mutexes_[index];

	if (!m) {
		mutexes_[index] = new Mutex;
		m = mutexes_[index];
	}

	return *m;
}
