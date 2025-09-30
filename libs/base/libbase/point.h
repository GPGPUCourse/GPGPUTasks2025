#pragma once

#include <array>
#include <stdexcept>

template <typename T>
class point2 {
public:
	T x, y;

	point2() : x(0), y(0) {}
	point2(T x, T y) : x(x), y(y) {}
	point2(const T* ptr) : x(ptr[0]), y(ptr[1]) {}
	template<typename T2>
	point2(const point2<T2> &that) : x(that.x), y(that.y) {}

	T& operator[](size_t index) {
		if (index == 0) return x;
		if (index == 1) return y;
		throw std::out_of_range("Index out of range");
	}

	const T& operator[](size_t index) const {
		if (index == 0) return x;
		if (index == 1) return y;
		throw std::out_of_range("Index out of range");
	}

	point2 operator+(const point2& other) const {
		return point2(x + other.x, y + other.y);
	}

	point2 operator-(const point2& other) const {
		return point2(x - other.x, y - other.y);
	}

	point2 operator*(const T& scalar) const {
		return point2(x * scalar, y * scalar);
	}

	point2 operator/(const T& scalar) const {
		return point2(x / scalar, y / scalar);
	}

	bool operator==(const point2& other) const {
		return x == other.x && y == other.y;
	}
};

template <typename T>
class point3 {
public:
	T x, y, z;

	point3() : x(0), y(0), z(0) {}
	point3(T x, T y, T z) : x(x), y(y), z(z) {}
	point3(const T* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]) {}
	template<typename T2>
	point3(const point3<T2> &that) : x(that.x), y(that.y), z(that.z) {}

	point2<T> xy()  const { return point2<T>(x, y);     }

	T& operator[](size_t index) {
		if (index == 0) return x;
		if (index == 1) return y;
		if (index == 2) return z;
		throw std::out_of_range("Index out of range");
	}

	const T& operator[](size_t index) const {
		if (index == 0) return x;
		if (index == 1) return y;
		if (index == 2) return z;
		throw std::out_of_range("Index out of range");
	}

	point3 operator+(const point3& other) const {
		return point3(x + other.x, y + other.y, z + other.z);
	}

	point3 operator-(const point3& other) const {
		return point3(x - other.x, y - other.y, z - other.z);
	}

	point3 operator*(const T& scalar) const {
		return point3(x * scalar, y * scalar, z * scalar);
	}

	point3 operator/(const T& scalar) const {
		return point3(x / scalar, y / scalar, z / scalar);
	}

	bool operator==(const point3& other) const {
		return x == other.x && y == other.y && z == other.z;
	}
};

template <typename T>
class point4 {
public:
	T x, y, z, w;

	point4() : x(0), y(0), z(0), w(0) {}
	point4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
	point4(const T* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]), w(ptr[3]) {}
	template<typename T2>
	point4(const point4<T2> &that) : x(that.x), y(that.y), z(that.z), w(that.w) {}

	point2<T> xy()  const { return point2<T>(x, y);     }
	point3<T> xyz() const { return point3<T>(x, y, z);  }

	T& operator[](size_t index) {
		if (index == 0) return x;
		if (index == 1) return y;
		if (index == 2) return z;
		if (index == 3) return w;
		throw std::out_of_range("Index out of range");
	}

	const T& operator[](size_t index) const {
		if (index == 0) return x;
		if (index == 1) return y;
		if (index == 2) return z;
		if (index == 3) return w;
		throw std::out_of_range("Index out of range");
	}

	point4 operator+(const point4& other) const {
		return point4(x + other.x, y + other.y, z + other.z, w + other.w);
	}

	point4 operator-(const point4& other) const {
		return point4(x - other.x, y - other.y, z - other.z, w - other.w);
	}

	point4 operator*(const T& scalar) const {
		return point4(x * scalar, y * scalar, z * scalar, w * scalar);
	}

	point4 operator/(const T& scalar) const {
		return point4(x / scalar, y / scalar, z / scalar, w / scalar);
	}

	bool operator==(const point4& other) const {
		return x == other.x && y == other.y && z == other.z && w == other.w;
	}
};

typedef point2<unsigned char>   point2uc;
typedef point4<unsigned char>   point4uc;
typedef point3<unsigned char>   point3uc;
typedef point2<unsigned int>    point2u;
typedef point4<unsigned int>    point4u;
typedef point3<unsigned int>    point3u;
typedef point2<int>             point2i;
typedef point4<int>             point4i;
typedef point3<int>             point3i;
typedef point2<float>           point2f;
typedef point3<float>           point3f;
typedef point4<float>           point4f;
typedef point2<double>          point2d;
typedef point3<double>          point3d;
typedef point4<double>          point4d;
