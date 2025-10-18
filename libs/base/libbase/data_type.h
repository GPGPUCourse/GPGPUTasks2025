#pragma once

#include "runtime_assert.h"

enum DataType {
	DataTypeUndefined,
	DataType8i,
	DataType8u,
	DataType16i,
	DataType16u,
	DataType32i,
	DataType32u,
        DataType16f,
	DataType32f,
	DataType64i,
	DataType64u,
	DataType64f
};

inline std::string typeName(DataType type) {
	if (type == DataTypeUndefined)  return "undefined";
	if (type == DataType8i)         return "8i";
	if (type == DataType8u)         return "8u";
	if (type == DataType16i)        return "16i";
	if (type == DataType16u)        return "16u";
	if (type == DataType32i)        return "32i";
	if (type == DataType32u)        return "32u";
	if (type == DataType16f)        return "16f";
	if (type == DataType32f)        return "32f";
	if (type == DataType64i)        return "64i";
	if (type == DataType64u)        return "64u";
	if (type == DataType64f)        return "64f";
	rassert(false, type, 9013891262);
}

inline void throwUnsupportedDataType(DataType data_type) {
	throw std::runtime_error("Unsupported data type: " + typeName(data_type));
}

inline size_t dataSize(DataType type) {
	rassert(type != DataTypeUndefined, 334721746);
	if (type == DataType8i  || type == DataType8u)                          return 1;
	if (type == DataType16i || type == DataType16u || type == DataType16f)  return 2;
	if (type == DataType32i || type == DataType32u || type == DataType32f)  return 4;
	if (type == DataType64i || type == DataType64u || type == DataType64f)  return 8;
	throwUnsupportedDataType(type);
        return 0; // fallback to silence -Wreturn-type
}

template <typename T>	class DataTypeTraits					{ public:	static DataType type() { return DataTypeUndefined;	} };
template<>				class DataTypeTraits<char>				{ public:	static DataType type() { return DataType8i;			} };
template<>				class DataTypeTraits<unsigned char>		{ public:	static DataType type() { return DataType8u;			} };
template<>				class DataTypeTraits<short>				{ public:	static DataType type() { return DataType16i;		} };
template<>				class DataTypeTraits<unsigned short>	{ public:	static DataType type() { return DataType16u;		} };
template<>				class DataTypeTraits<int>				{ public:	static DataType type() { return DataType32i;		} };
template<>				class DataTypeTraits<unsigned int>		{ public:	static DataType type() { return DataType32u;		} };
template<>				class DataTypeTraits<float>				{ public:	static DataType type() { return DataType32f;		} };
template<>				class DataTypeTraits<long long>			{ public:	static DataType type() { return DataType64i;		} };
template<>				class DataTypeTraits<unsigned long long>{ public:	static DataType type() { return DataType64u;		} };
template<>				class DataTypeTraits<double>			{ public:	static DataType type() { return DataType64f;		} };
