#include "dtype.h"
#include <iostream>
#include <cstring>
#include <cmath>

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

void* initialize_data(double value, DType dtype) {
  void* data = malloc(dtype_size(dtype));
  if (!data) {
    std::cerr << "Memory allocation failed!" << std::endl;
    return nullptr;
  }

  set_data_from_double(data, dtype, value);
  return data;
}

void convert_data(void* data, DType from_dtype, DType to_dtype) {
  double value = get_data_as_double(data, from_dtype, 0);
  set_data_from_double(data, to_dtype, value);
}

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

double get_data_as_double(void* data, DType dtype, int index) {
  switch (dtype) {
    case DType::INT8:
      return static_cast<double>(reinterpret_cast<int8_t*>(data)[index]);
    case DType::INT16:
      return static_cast<double>(reinterpret_cast<int16_t*>(data)[index]);
    case DType::INT32:
      return static_cast<double>(reinterpret_cast<int32_t*>(data)[index]);
    case DType::INT64:
      return static_cast<double>(reinterpret_cast<int64_t*>(data)[index]);
    case DType::FLOAT32:
      return static_cast<double>(reinterpret_cast<float*>(data)[index]);
    case DType::FLOAT64:
      return reinterpret_cast<double*>(data)[index];
    default:
      std::cerr << "Error: Unsupported data type." << std::endl;
      return 0.0;
  }
}

void set_data_from_double(void* data, DType dtype, double value) {
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
      *reinterpret_cast<float*>(data) = static_cast<float>(value);
      break;
    case DType::FLOAT64:
      *reinterpret_cast<double*>(data) = value;
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