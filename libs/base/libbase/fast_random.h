#pragma once

#include <limits>
#include <libbase/point.h>

// See https://stackoverflow.com/a/1640399
class FastRandom {
public:
    FastRandom(unsigned long seed=123456789) {
        reset(seed);
    }

    void reset(unsigned long seed=123456789) {
        x = seed;
        y = 362436069;
        z = 521288629;
    }

    // Returns pseudo-random value in range [min; max] (inclusive)
    int next(int min=0, int max=std::numeric_limits<int>::max()) {
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

        unsigned long t = x;
        x = y;
        y = z;
        z = t ^ x ^ y;

        return min + (unsigned int) (z % (((unsigned long) max) - min + 1));
    }

    float nextf() {
        return (next() * 2000.0f / std::numeric_limits<int>::max()) - 1000.0f;
    }

    point3uc nextColor() {
        return point3uc(next(0, 255), next(0, 255), next(0, 255));
    }

private:
    unsigned long x, y, z;
};
