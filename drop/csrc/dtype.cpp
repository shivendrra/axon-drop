/*
  - dtype.cpp main file that contains all dtype related ops
  - changes the dtype of each scalar & tensor values, up-castes & re-castes
    values: float->dtype->float as needed.
  - tested; works fine.
*/

#include "dtype.h"
#include <iostream>
#include <cstring>
#include <cmath>

// returns the size of the given data type
size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::INT8: return sizeof(int8_t);
    case DType::INT16: return sizeof(int16_t);
    case DType::INT32: return sizeof(int32_t);
    case DType::INT64: return sizeof(int64_t);
    case DType::FLOAT32: return sizeof(float);
    case DType::FLOAT64: return sizeof(double);
    default: return 0;
  }
}

// initializes a memory block for the given value and dtype
void* initialize_data(double value, DType dtype) {
  void* data = malloc(dtype_size(dtype));
  if (!data) {
    std::cerr << "Memory allocation failed!" << std::endl;
    return nullptr;
  }
  set_data_from_float(data, dtype, value);
  return data;
}

// converts data from one dtype to another
void convert_data(void* data, DType from_dtype, DType to_dtype) {
  double value = get_data_as_float(data, from_dtype, 0);
  set_data_from_float(data, to_dtype, value);
}

// converts dtype to string for display
std::string dtype_to_string(DType dtype) {
  switch (dtype) {
    case DType::INT8: return "INT8";
    case DType::INT16: return "INT16";
    case DType::INT32: return "INT32";
    case DType::INT64: return "INT64";
    case DType::FLOAT32: return "FLOAT32";
    case DType::FLOAT64: return "FLOAT64";
    default: return "Unknown";
  }
}

// retrieves data as double from given index & dtype
float get_data_as_float(void* data, DType dtype, int index) {
  float result = 0.0;
  size_t type_size = dtype_size(dtype);
  char* raw_data = static_cast<char*>(data) + index * type_size;
  switch (dtype) {
    case DType::INT8:
      result = static_cast<float>(*reinterpret_cast<int8_t*>(raw_data));
      break;
    case DType::INT16:
      result = static_cast<float>(*reinterpret_cast<int16_t*>(raw_data));
      break;
    case DType::INT32:
      result = static_cast<float>(*reinterpret_cast<int32_t*>(raw_data));
      break;
    case DType::INT64:
      result = static_cast<float>(*reinterpret_cast<int64_t*>(raw_data));
      break;
    case DType::FLOAT32:
      result = static_cast<float>(*reinterpret_cast<float*>(raw_data));
      break;
    case DType::FLOAT64:
      result = *reinterpret_cast<float*>(raw_data);
      break;
    default:
      std::cerr << "Error: Unsupported data type." << std::endl;
      return 0.0;
  }
  return result;
}

// sets data from & double value based on dtype
void set_data_from_float(void* data, DType dtype, float value) {
  switch (dtype) {
    case DType::INT8:
      *reinterpret_cast<int8_t*>(data) = static_cast<int8_t>(std::round(value));
      break;
    case DType::INT16:
      *reinterpret_cast<int16_t*>(data) = static_cast<int16_t>(std::round(value));
      break;
    case DType::INT32:
      *reinterpret_cast<int32_t*>(data) = static_cast<int32_t>(std::round(value));
      break;
    case DType::INT64:
      *reinterpret_cast<int64_t*>(data) = static_cast<int64_t>(std::round(value));
      break;
    case DType::FLOAT32:
      *reinterpret_cast<float*>(data) = value;
      break;
    case DType::FLOAT64:
      *reinterpret_cast<double*>(data) = static_cast<float>(value);
      break;
    default:
      std::cerr << "Unknown dtype!" << std::endl;
  }
}

void free_data(void* data) {
  if (data) {
    free(data);
  }
}