#ifndef __DTYPE_H__
#define __DTYPE_H__

#include <cstdint>
#include <stdint.h>

enum class DType {
  INT8,
  INT16,
  INT32,
  INT64,
  FLOAT32,
  FLOAT64
};

size_t dtype_size(DType dtype);
void* initialize_data(float value, DType dtype);
void convert_data(void* data, DType from_dtype, DType to_dtype);
const char* dtype_to_string(DType dtype);
float get_data_as_float(void* data, DType dtype);
void set_data_from_float(void* data, DType dtype, float value);
void free_data(void* data);

#endif