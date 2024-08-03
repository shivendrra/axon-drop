def get_shape(data:list) -> list:
  if isinstance(data, list):
    return [len(data), ] + get_shape(data[0])
  else:
    return []