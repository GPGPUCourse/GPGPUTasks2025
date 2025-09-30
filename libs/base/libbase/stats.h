#pragma once

#include <libbase/runtime_assert.h>

#include <vector>
#include <string>
#include <ostream>
#include <sstream>
#include <algorithm>

namespace stats {

	template<typename T>
	T min(const std::vector<T> &values) {
		if (values.size() == 0)
			return (T) 0;

		auto res = min_element(std::begin(values), std::end(values));
		return *res;
	}

	template<typename T>
	T avg(const std::vector<T> &values) {
		T sum = 0;
		for (size_t i = 0; i < values.size(); ++i) {
			sum += values[i];
		}
		T avg;
		if (values.size() > 0) {
			avg = sum / values.size();
		} else {
			avg = sum;
		}
		return avg;
	}

	template<typename T>
	T percentile(const std::vector<T> &values, int percentile) {
		rassert(percentile >= 0 && percentile <= 100, 4675356245421);
		if (values.size() == 0)
			return (T) 0;

		std::vector<T> copy = values;
		size_t m = values.size() * percentile / 100;
		std::nth_element(copy.begin(), copy.begin() + m, copy.end());
		return copy[m];
	}

	template<typename T>
	T median(const std::vector<T> &values) {
		return percentile(values, 50);
	}

	template<typename T>
	T max(const std::vector<T> &values) {
		if (values.size() == 0)
			return (T) 0;

		auto res = max_element(std::begin(values), std::end(values));
		return *res;
	}

	template<typename T>
	std::string valuesStatsLine(const std::vector<T> &values) {
		std::ostringstream ss;
		T min = stats::min(values);
		T perc10 = percentile(values, 10);
		T perc50 = percentile(values, 50);
		T perc90 = percentile(values, 90);
		T max = stats::max(values);

		ss << values.size() << " values (min=" << min << " 10%=" << perc10 << " median=" << perc50 << " 90%=" << perc90 << " max=" << max << ")";
		return ss.str();
	}

	template<typename T>
	std::string vectorToString(const std::vector<T> &values, size_t printed_elements_limit=10)
	{
		std::ostringstream ss;

		size_t n = values.size();
		ss << n << " values: [";
		if (n <= printed_elements_limit) {
			for (size_t i = 0; i < n; ++i) {
				ss << values[i] << (i + 1 < n ? " " : "");
			}
		} else {
			size_t nelements_prefix = printed_elements_limit / 2;
			for (size_t i = 0; i < nelements_prefix; ++i) {
				ss << values[i] << " ";
			}
			ss << "...";
			size_t nelements_suffix = printed_elements_limit - nelements_prefix;
			for (size_t i = n - nelements_suffix; i < n; ++i) {
				ss << " " << values[i];
			}
		}
		ss << "]";
		return ss.str();
	}

}
