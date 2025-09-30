#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <cstdint>

template <typename T>
std::string to_string(T value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

std::vector<std::string> split(const std::string &string, const std::string &separator, bool keep_empty_parts = true);
size_t replace_all(std::string &str, const std::string &old_str, const std::string &new_str);
std::string join(const std::vector<std::string> &tokens, const std::string &separator);
std::istream &getline(std::istream &is, std::string &str);
double atof(const std::string &s);
int atoi(const std::string &s);
std::string tolower(const std::string &str);
std::string toupper(const std::string &str);
std::string trimmed(const std::string &str);
std::string base64_encode(const std::string &in);
std::string base64_decode(const std::string &in);
bool starts_with(const std::string &str, const std::string &prefix);
bool ends_with(const std::string &str, const std::string &suffix);

std::string format(const std::string &s, const std::string &arg1);

std::string to_string_pad_zeros(int64_t value, uint64_t padding = 5);

inline std::string format(const std::string &s)
{
	return s;
}

template<typename T, typename... TS>
std::string format(const std::string &s, const T &arg, const TS&... args)
{
	return format(format(s, to_string(arg)), args...);
}

template <typename T>
std::string to_percent(T part, T total)
{
	if (total == 0)
		return "0%";
	return to_string((int) std::floor(part * 100.0 / total + 0.5)) + "%";
}

template <typename T>
std::string total_percent(const std::vector<T> &parts, T total)
{
	int sum = 0;
	for (T part : parts) {
		sum += (int) std::floor(part * 100.0 / total + 0.5);
	}
	return to_string(sum) + "%";
}
