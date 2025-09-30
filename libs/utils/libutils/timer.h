#pragma once

#ifdef _WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif

#include <vector>
#include <cmath>
#include <algorithm>

class timer {
protected:
#ifdef _WIN32
    typedef clock_t timer_type;
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
        if (is_running_) return;

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
        timer_type tm;
#ifdef _WIN32
        tm = clock();
#else
        ::gettimeofday(&tm, 0);
#endif
        return tm;
    }

    static double diff(const timer_type &start, const timer_type &end)
    {
#ifdef _WIN32
        return (double) (end - start) / (double) CLOCKS_PER_SEC;
#else
        long secs = end.tv_sec - start.tv_sec;
        long usecs = end.tv_usec - start.tv_usec;

        return (double) secs + (double) usecs / 1000000.0;
#endif
    }
};
