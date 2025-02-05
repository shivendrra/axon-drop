def get_shape(data):
  if isinstance(data, list):
    return [len(data), ] + get_shape(data[0])
  return []

def flatten(data):
  if isinstance(data, list):
    return [item for sublist in data for item in flatten(sublist)]
  return [data]

def get_strides(shape):
  ndim = len(shape)
  strides = [0] * ndim
  stride = 1
  for i in range(ndim - 1, -1, -1):
    strides[i] = stride
    stride *= shape[i]  
  return strides

def flatten_recursive(data:list, start_dim:int=0, end_dim:int=-1) -> list:
  def _recurse_flatten(data, current_dim):
    if current_dim < start_dim:
      return [_recurse_flatten(item, current_dim + 1) for item in data]
    elif start_dim <= current_dim <= end_dim:
      return flatten(data)
    else:
      return data
  if end_dim == -1:
    end_dim = len(get_shape(data)) - 1
  return _recurse_flatten(data, 0)

def reshape(data:list, new_shape:tuple) -> list:
  assert type(new_shape) == tuple, "new shape must be a tuple"
  def _shape_numel(shape):
    numel = 1
    for dim in shape:
      numel *= dim
    return numel

  def unflatten(flat, shape):
    if len(shape) == 1:
      return flat[:shape[0]]
    size = shape[0]
    return [unflatten(flat[i*int(len(flat)/size):(i+1)*int(len(flat)/size)], shape[1:]) for i in range(size)]

  def infer_shape(shape, total_size):
    if shape.count(-1) > 1:
      raise ValueError("Only one dimension can be -1")
    unknown_dim, known_dims, known_size = shape.index(-1) if -1 in shape else None, [dim for dim in shape if dim != -1], 1
    for dim in known_dims:
      known_size *= dim      
    if unknown_dim is not None:
      inferred_size = total_size // known_size
      if inferred_size * known_size != total_size:
        raise ValueError(f"Cannot reshape array to shape {shape}")
      shape = list(shape)
      shape[unknown_dim] = inferred_size
    return shape
  original_size = _shape_numel(get_shape(data))
  new_shape, new_size = infer_shape(new_shape, original_size), _shape_numel(new_shape)
  if original_size != new_size:
    raise ValueError(f"Cannot reshape array of size {original_size} to shape {new_shape}")
  flat_data = flatten(data)
  return unflatten(flat_data, new_shape)

def _zeros(shape):
  if not shape:
    return [0]
  if len(shape) == 1:
    return [0] * shape[0]
  return [_zeros(shape[1:]) for _ in range(shape[0])]

def can_broadcast(shape1, shape2) -> bool:
  len1, len2 = len(shape1), len(shape2)
  if len1 < len2:
    shape1 = [1] * (len2 - len1) + shape1
  elif len2 < len1:
    shape2 = [1] * (len1 - len2) + shape2
  for dim1, dim2 in zip(shape1, shape2):
    if dim1 != dim2 and dim1 != 1 and dim2 != 1:
      return False
  return True