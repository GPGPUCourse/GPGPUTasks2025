#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

#ifdef _WIN32
  #define NOMINMAX
  #include <windows.h>     // QueryPerformanceCounter / Frequency
#else
  #include <time.h>        // clock_gettime / timespec
  #include <sys/time.h>    // gettimeofday fallback
  #ifdef __APPLE__
    #include <mach/mach_time.h> // mach_absolute_time
  #endif
#endif

class timer {
protected:
#ifdef _WIN32
    // Use QPC ticks for high-resolution monotonic timing on Windows
    typedef long long timer_type;
#elif defined(__APPLE__)
    // Use mach_absolute_time ticks (monotonic, high-res) on macOS
    typedef unsigned long long timer_type;
#elif defined(CLOCK_MONOTONIC)
    // Use timespec with CLOCK_MONOTONIC on POSIX
    typedef struct timespec timer_type;
#else
    typedef struct timeval timer_type;
#endif

    double counter_;
    timer_type start_;
    int is_running_;

    std::vector<double> laps_;

public:
    timer(bool paused = false)
    {
        counter_ = 0;
        is_running_ = 0;
        if (!paused)
            start();
    }

    void start()
    {
        if (is_running_) restart();

        start_ = measure();
        is_running_ = 1;
    }

    void stop()
    {
        if (!is_running_) return;

        counter_ += diff(start_, measure());
        is_running_ = 0;
    }

    double nextLap()
    {
        double lap_time = elapsed();
        laps_.push_back(lap_time);
        restart();
        return lap_time;
    }

    void reset()
    {
        counter_ = 0;
        is_running_ = 0;
    }

    void restart()
    {
        reset();
        start();
    }

    double elapsed() const
    {
        double tm = counter_;

        if (is_running_)
            tm += diff(start_, measure());

        if (tm < 0)
            tm = 0;

        return tm;
    }

    const std::vector<double>& laps() const
    {
        return laps_;
    }

    // Note that this is not true averaging, if there is at least 5 laps - averaging made from 20% percentile to 80% percentile (See lapsFiltered)
    double lapAvg() const
    {
        std::vector<double> laps = lapsFiltered();

        double sum = 0.0;
        for (int i = 0; i < laps.size(); ++i) {
            sum += laps[i];
        }
        if (laps.size() > 0) {
            sum /= laps.size();
        }
        return sum;
    }

    // Note that this is not true averaging, if there is at least 5 laps - averaging made from 20% percentile to 80% percentile (See lapsFiltered)
    double lapStd() const
    {
        double avg = lapAvg();

        std::vector<double> laps = lapsFiltered();

        double sum2 = 0.0;
        for (int i = 0; i < laps.size(); ++i) {
            sum2 += laps[i] * laps[i];
        }
        if (laps.size() > 0) {
            sum2 /= laps.size();
        }
        return sqrt(std::max(0.0, sum2 - avg * avg));
    }

protected:

    std::vector<double> lapsFiltered() const
    {
        std::vector<double> laps = laps_;
        std::sort(laps.begin(), laps.end());

        unsigned int nlaps = laps.size();
        if (nlaps >= 5) {
            // Removing last 20% of measures
            laps.erase(laps.end() - nlaps/5, laps.end());
            // Removing first 20% of measures
            laps.erase(laps.begin(), laps.begin() + nlaps/5);
        }
        return laps;
    }

    static timer_type measure()
    {
#ifdef _WIN32
        // QPC is monotonic and high resolution
        LARGE_INTEGER v;
        ::QueryPerformanceCounter(&v);
        return (timer_type)v.QuadPart;

#elif defined(__APPLE__)
        // mach_absolute_time is monotonic with nanosecond scale via timebase
        return (timer_type)::mach_absolute_time();

#elif defined(CLOCK_MONOTONIC)
        // clock_gettime(CLOCK_MONOTONIC) typically uses vDSO (very low overhead)
        struct timespec ts;
        ::clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts;

#else
        // Fallback to gettimeofday (wall clock, not monotonic) if nothing else is available
        struct timeval tv;
        ::gettimeofday(&tv, 0);
        return tv;
#endif
    }

    static double diff(const timer_type &start, const timer_type &end)
    {
#ifdef _WIN32
        // Convert QPC ticks to seconds using cached frequency
        const double inv_freq = qpc_inv_freq();
        const long long dt = (long long)(end - start);
        return (double)dt * inv_freq;

#elif defined(__APPLE__)
        // Convert mach ticks to seconds using cached timebase
        const mach_timebase_info_data_t& tb = timebase();
        // Convert ticks -> nanoseconds: ticks * numer/denom
        const unsigned long long dt = (unsigned long long)(end - start);
        const long double ns = (long double)dt * (long double)tb.numer / (long double)tb.denom;
        return (double)(ns * 1e-9L);

#elif defined(CLOCK_MONOTONIC)
        // timespec difference in seconds (monotonic)
        long sec  = end.tv_sec  - start.tv_sec;
        long nsec = end.tv_nsec - start.tv_nsec;
        if (nsec < 0) { --sec; nsec += 1000000000L; }
        return (double)sec + (double)nsec * 1e-9;

#else
        // timeval fallback (microsecond precision, not monotonic)
        long secs  = end.tv_sec  - start.tv_sec;
        long usecs = end.tv_usec - start.tv_usec;
        return (double)secs + (double)usecs / 1000000.0;
#endif
    }

#ifdef _WIN32
    // Cache 1/frequency to avoid QueryPerformanceFrequency in hot path
    static double qpc_inv_freq()
    {
        static double inv = []{
            LARGE_INTEGER f;
            ::QueryPerformanceFrequency(&f);
            return 1.0 / (double)f.QuadPart;
        }();
        return inv;
    }
#endif

#ifdef __APPLE__
    // Cache timebase to avoid repeated syscalls
    static const mach_timebase_info_data_t& timebase()
    {
        static mach_timebase_info_data_t tb = []{
            mach_timebase_info_data_t info{};
            ::mach_timebase_info(&info);
            return info;
        }();
        return tb;
    }
#endif
};
