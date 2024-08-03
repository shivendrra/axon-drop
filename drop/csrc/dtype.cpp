#include "dtype.h"
#include <cstring>
#include <iostream>
#include <stdexcept>

void* initialize_data(double value, DType dtype) {
  void* data = nullptr;
  switch(dtype) {
    case DType::INT8: {
      data = malloc(sizeof(int8_t));
      int8_t int8_val = static_cast<int8_t>(value);
      memcpy(data, &int8_val, sizeof(int8_t));
      break;
    }
    case DType::INT16: {
      data = malloc(sizeof(int16_t));
      int16_t int16_val = static_cast<int16_t>(value);
      memcpy(data, &int16_val, sizeof(int16_t));
      break;
    }
    case DType::INT32: {
      data = malloc(sizeof(int32_t));
      int32_t int32_val = static_cast<int32_t>(value);
      memcpy(data, &int32_val, sizeof(int32_t));
      break;
    }
    case DType::INT64: {
      data = malloc(sizeof(int64_t));
      int64_t int64_val = static_cast<int64_t>(value);
      memcpy(data, &int64_val, sizeof(int64_t));
      break;
    }
    case DType::FLOAT32: {
      data = malloc(sizeof(float));
      float float32_val = static_cast<float>(value);
      memcpy(data, &float32_val, sizeof(float));
      break;
    }
    case DType::FLOAT64: {
      data = malloc(sizeof(double));
      double float64_val = static_cast<double>(value);
      memcpy(data, &float64_val, sizeof(double));
      break;
    }
    default:
      throw std::invalid_argument("Unsupported dtype");
  }
  return data;
}

void convert_data(void* data, DType from_dtype, DType to_dtype) {
  if (from_dtype == to_dtype) return;
  double value = get_data_as_double(data, from_dtype);
  set_data_from_double(data, to_dtype, value);
}

double get_data_as_double(void* data, DType dtype) {
  switch(dtype) {
    case DType::INT8: return static_cast<double>(*reinterpret_cast<int8_t*>(data));
    case DType::INT16: return static_cast<double>(*reinterpret_cast<int16_t*>(data));
    case DType::INT32: return static_cast<double>(*reinterpret_cast<int32_t*>(data));
    case DType::INT64: return static_cast<double>(*reinterpret_cast<int64_t*>(data));
    case DType::FLOAT32: return static_cast<double>(*reinterpret_cast<float*>(data));
    case DType::FLOAT64: return *reinterpret_cast<double*>(data);
    default: throw std::invalid_argument("Unsupported dtype");
  }
}

void set_data_from_double(void* data, DType dtype, double value) {
  switch(dtype) {
    case DType::INT8: {
      int8_t int8_val = static_cast<int8_t>(value);
      memcpy(data, &int8_val, sizeof(int8_t));
      break;
    }
    case DType::INT16: {
      int16_t int16_val = static_cast<int16_t>(value);
      memcpy(data, &int16_val, sizeof(int16_t));
      break;
    }
    case DType::INT32: {
      int32_t int32_val = static_cast<int32_t>(value);
      memcpy(data, &int32_val, sizeof(int32_t));
      break;
    }
    case DType::INT64: {
      int64_t int64_val = static_cast<int64_t>(value);
      memcpy(data, &int64_val, sizeof(int64_t));
      break;
    }
    case DType::FLOAT32: {
      float float32_val = static_cast<float>(value);
      memcpy(data, &float32_val, sizeof(float));
      break;
    }
    case DType::FLOAT64: {
      double float64_val = static_cast<double>(value);
      memcpy(data, &float64_val, sizeof(double));
      break;
    }
    default: throw std::invalid_argument("Unsupported dtype");
  }
}

void free_data(void* data) {
  free(data);
}

std::string dtype_to_string(DType dtype) {
  switch(dtype) {
    case DType::INT8: return "INT8";
    case DType::INT16: return "INT16";
    case DType::INT32: return "INT32";
    case DType::INT64: return "INT64";
    case DType::FLOAT32: return "FLOAT32";
    case DType::FLOAT64: return "FLOAT64";
    default: return "UNKNOWN";
  }
}