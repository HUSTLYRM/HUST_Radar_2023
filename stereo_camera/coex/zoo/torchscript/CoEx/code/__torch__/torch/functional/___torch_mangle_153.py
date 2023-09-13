def norm(input: Tensor,
    p: Optional[number],
    dim: Optional[int]=None,
    keepdim: bool=False,
    out: Optional[Tensor]=None,
    dtype: Optional[int]=None) -> Tensor:
  ndim = torch.dim(input)
  if torch.__is__(dim, None):
    _0, dim0 = torch.__is__(out, None), dim
  else:
    _0, dim0 = False, unchecked_cast(int, dim)
  if _0:
    _1 = torch.__is__(dtype, None)
  else:
    _1 = False
  if _1:
    _2 = torch.__isnot__(p, None)
  else:
    _2 = False
  if _2:
    p0 = unchecked_cast(number, p)
    _dim = annotate(List[int], [])
    for i in range(ndim):
      _4 = torch.append(_dim, i)
    _5 = torch.norm(input, p0, _dim, keepdim)
    _3 = _5
  else:
    if torch.__isnot__(dim0, None):
      dim1 = unchecked_cast(int, dim0)
      dim2 = unchecked_cast(int, dim1)
      _dim0 = [dim2]
    else:
      _dim0 = None
    if torch.__is__(_dim0, None):
      _dim2 = annotate(List[int], [])
      for _6 in range(ndim):
        _7 = torch.append(_dim2, _6)
      _dim1 = _dim2
    else:
      _dim1 = unchecked_cast(List[int], _dim0)
    if torch.__is__(out, None):
      if torch.__is__(dtype, None):
        _10 = torch.norm(input, p, _dim1, keepdim)
        _9 = _10
      else:
        dtype0 = unchecked_cast(int, dtype)
        _11 = torch.norm(input, p, _dim1, keepdim, dtype=dtype0)
        _9 = _11
      _8 = _9
    else:
      out0 = unchecked_cast(Tensor, out)
      if torch.__is__(dtype, None):
        _13 = torch.norm(input, p, _dim1, keepdim, out=out0)
        _12 = _13
      else:
        dtype1 = unchecked_cast(int, dtype)
        _14 = torch.norm(input, p, _dim1, keepdim, dtype=dtype1, out=out0)
        _12 = _14
      _8 = _12
    _3 = _8
  return _3
