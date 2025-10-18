#include "string_utils.h"

#include <iomanip>
#include <sstream>

std::vector<std::string> split(const std::string &string, const std::string &separator, bool keep_empty_parts)
{
	std::vector<std::string> result;
	size_t p = 0;
	
	while (true) {
		size_t s = string.find(separator, p);
		if (s == std::string::npos)
			break;
		std::string token = string.substr(p, s - p);
		if (keep_empty_parts || token.size())
			result.push_back(token);
		p = s + separator.size();
	}

	std::string token = string.substr(p);
	if (keep_empty_parts || token.size())
		result.push_back(token);
	return result;
}

std::string join(const std::vector<std::string> &tokens, const std::string &separator)
{
	std::string res;
	for (size_t i = 0; i < tokens.size(); i++) {
		if (i)
			res += separator;
		res += tokens[i];
	}
	return res;
}

std::istream &getline(std::istream &is, std::string &str)
{
	std::string::size_type nread = 0;

	if (std::istream::sentry(is, true)) {
		std::streambuf *const sbuf = is.rdbuf();
		str.clear();

		while (nread < str.max_size()) {
			int c1 = sbuf->sbumpc();
			if (c1 == std::streambuf::traits_type::eof()) {
				is.setstate(std::istream::eofbit);
				break;
			} else {
				++nread;
				const char ch = c1;
				if (ch != '\n' && ch != '\r') {
					str.push_back(ch);
				} else {
					const char ch1 = is.peek();
					if (ch == '\n' && ch1 == '\r') is.ignore(1);
					if (ch == '\r' && ch1 == '\n') is.ignore(1);
					break;
				}
			}
		}
	}

	if (nread == 0 || nread >= str.max_size()) {
		is.setstate(std::istream::failbit);
	}

	return is;
}

double atof(const std::string &s)
{
	std::stringstream ss(s);
	ss.imbue(std::locale::classic());

	double value = 0;
	ss >> value;
	return value;
}

int atoi(const std::string &s)
{
	std::stringstream ss(s);
	ss.imbue(std::locale::classic());

	int value = 0;
	ss >> value;
	return value;
}

std::string tolower(const std::string &str)
{
	std::string res = str;
	for (size_t k = 0; k < res.size(); k++) res[k] = ::tolower(res[k]);
	return res;
}

std::string toupper(const std::string &str)
{
	std::string res = str;
	for (size_t k = 0; k < res.size(); k++) res[k] = ::toupper(res[k]);
	return res;
}

std::string trimmed(const std::string &s)
{
	const size_t p1 = s.find_first_not_of(' ');
	const size_t p2 = s.find_last_not_of(' ');

	if (p1 == std::string::npos)
		return std::string();

	return s.substr(p1, p2 - p1 + 1);
}

// base 64 encoding/decoding
// http://stackoverflow.com/questions/180947/base64-decode-snippet-in-c

std::string base64_encode(const std::string &in)
{
	std::string out;

	int val=0, valb=-6;
	for (std::string::const_iterator it = in.begin(); it != in.end(); ++it) {
		unsigned char c = *it;

		val = (val<<8) + c;
		valb += 8;
		while (valb>=0) {
			out.push_back("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[(val>>valb)&0x3F]);
			valb-=6;
		}
	}

	if (valb>-6) out.push_back("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[((val<<8)>>(valb+8))&0x3F]);
	while (out.size()%4) out.push_back('=');
	return out;
}

std::string base64_decode(const std::string &in)
{
	std::string out;

	std::vector<int> T(256,-1);
	for (int i=0; i<64; i++) T["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i]] = i; 

	int val=0, valb=-8;
	for (std::string::const_iterator it = in.begin(); it != in.end(); ++it) {
		unsigned char c = *it;
		if (isspace(c))
			continue;

		if (T[c] == -1)
			break;

		val = (val<<6) + T[c];
		valb += 6;
		if (valb>=0) {
			out.push_back(char((val>>valb)&0xFF));
			valb-=8;
		}
	}

	return out;
}

bool starts_with(const std::string &str, const std::string &prefix)
{
	if (prefix.size() > str.size())
		return false;

	return str.substr(0, prefix.size()) == prefix;
}

bool ends_with(const std::string &str, const std::string &suffix)
{
	if (suffix.size() > str.size())
		return false;

	return str.substr(str.size() - suffix.size()) == suffix;
}

size_t replace_all(std::string &str, const std::string &old_str, const std::string &new_str)
{
	size_t count = 0;

	size_t pos = 0;
	while((pos = str.find(old_str, pos)) != std::string::npos) {
		str.replace(pos, old_str.length(), new_str);
		pos += new_str.length();
		count++;
	}

	return count;
}

std::string format(const std::string &s, const std::string &arg1)
{
	std::string res = s;
	replace_all(res, "%1", arg1);
	for (size_t i = 1; i < res.size(); i++) {
		if (res[i - 1] != '%')
			continue;
		if (res[i] >= '2' && res[i] <= '9')
			res[i] = res[i] - 1;
	}
	return res;
}

std::string to_string_pad_zeros(int64_t value, uint64_t padding)
{
	std::stringstream out;
	out << std::setfill('0') << std::setw(padding) << value;
	return out.str();
}
