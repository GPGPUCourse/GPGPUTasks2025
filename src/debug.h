#include <libbase/runtime_assert.h>

#include <bitset>
#include <vector>
#include <string>
#include <algorithm>
#include <climits>
#include <type_traits>


namespace debug {
    std::vector<int> toInt(const std::vector<unsigned char> &values) {
        std::vector<int> res(values.size());

        for (unsigned int i = 0; i < values.size(); ++i) {
            res[i] = values[i];
        }

        return res;
    }

    template<class T>
    std::vector<std::string> toBits(const std::vector<T> &values) {
        static_assert(std::is_integral<T>::value, "toBits<T> requires integral T");
        using U = typename std::make_unsigned<T>::type;
        constexpr size_t N = sizeof(T) * CHAR_BIT;

        std::vector<std::string> res(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            std::bitset<N> bs(static_cast<U>(values[i]));
            res[i] = bs.to_string();
        }

        return res;
    }

    std::vector<std::string> highlightBits(const std::vector<std::string> &values, unsigned int bit_offset, unsigned int nbits) {
        std::vector<std::string> res(values.size());
        for (unsigned int i = 0; i < values.size(); ++i) {
            std::string bits = values[i];
            int bits_count = bits.size();
            int bits_to = bits_count - bit_offset;
            rassert(bits_to > 0, bits_to, 546234132312);
            int bits_from = std::max(0, bits_to - (int) nbits);
            res[i] = bits.substr(0, bits_from) + "[" + bits.substr(bits_from, bits_to - bits_from) + "]" + bits.substr(bits_to, bits.size() - bits_to);
        }
        return res;
    }

    std::vector<std::string> cutLeadingBits(const std::vector<std::string> &values, unsigned int nbits) {
        std::vector<std::string> res(values.size());
        for (unsigned int i = 0; i < values.size(); ++i) {
            std::string bits = values[i];
            rassert(bits.size() > nbits, 546343112312);
            res[i] = bits.substr(nbits);
        }
        return res;
    }

    int countLeadingZeroBits(unsigned value) {
        if (value == 0)
            return 32;

        int clz = 0;
        while (clz < 31 && (value & (0xffffffffu >> (clz + 1))) == value) {
            ++clz;
        }
        return clz;
    }

    std::vector<std::string> prettyBits(const std::vector<unsigned int> &values, unsigned int max_value, unsigned int bit_offset, unsigned int nbits) {
        // these are like unit-tests but a bit more reliable & insane (it runs each time before each real function execution ahahah)
        rassert(countLeadingZeroBits(0) == 32, 54624345234);
        rassert(countLeadingZeroBits(1) == 31, 54624345235);
        rassert(countLeadingZeroBits(2) == 30, 54624345236);
        rassert(countLeadingZeroBits(3) == 30, 54624345237);
        rassert(countLeadingZeroBits(4) == 29, 54624345238);
        rassert(countLeadingZeroBits(8) == 28, 54624345239);
        unsigned int meaninglessBitsCount = countLeadingZeroBits(max_value); // count number of leading zero bits

        auto fullBits = toBits(values);
        auto shortBits = cutLeadingBits(fullBits, meaninglessBitsCount);
        auto highlightedBits = highlightBits(shortBits, bit_offset, nbits);
        return highlightedBits;
    }
}