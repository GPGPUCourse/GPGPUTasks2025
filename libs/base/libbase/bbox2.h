#pragma once

#include "point.h"


template <typename T>
class bbox2 {
private:
	point2<T> minPoint;
	point2<T> maxPoint;

public:
	void clear() {
		minPoint = point2<T>(std::numeric_limits<T>::max(), std::numeric_limits<T>::max());
		maxPoint = point2<T>(0, 0);
		rassert(isEmpty(), 443460392);
	}

	bool isEmpty() const {
		return minPoint.x > maxPoint.x || minPoint.y > maxPoint.y;
	}

	bbox2() {
		clear();
	}

	void grow(const point2<T>& point) {
		if (isEmpty()) {
			minPoint = maxPoint = point;
		} else {
			minPoint.x = std::min(minPoint.x, point.x);
			minPoint.y = std::min(minPoint.y, point.y);
			maxPoint.x = std::max(maxPoint.x, point.x);
			maxPoint.y = std::max(maxPoint.y, point.y);
		}
	}

	T width() const {
		return isEmpty() ? 0 : maxPoint.x - minPoint.x;
	}

	T height() const {
		return isEmpty() ? 0 : maxPoint.y - minPoint.y;
	}

	void grow(const bbox2<T>& other) {
		if (other.isEmpty()) return;

		if (isEmpty()) {
			minPoint = other.minPoint;
			maxPoint = other.maxPoint;
		} else {
			minPoint.x = std::min(minPoint.x, other.minPoint.x);
			minPoint.y = std::min(minPoint.y, other.minPoint.y);
			maxPoint.x = std::max(maxPoint.x, other.maxPoint.x);
			maxPoint.y = std::max(maxPoint.y, other.maxPoint.y);
		}
	}

	bool contains(const point2<T>& point) const {
		return !isEmpty() &&
		       point.x >= minPoint.x && point.x <= maxPoint.x &&
		       point.y >= minPoint.y && point.y <= maxPoint.y;
	}

	point2<T> size() const {
		return point2<T>(width(), height());
	}

	point2<T> min() const {
		return minPoint;
	}

	point2<T> max() const {
		return maxPoint;
	}
};

typedef bbox2<float>    bbox2f;
typedef bbox2<double>   bbox2d;
