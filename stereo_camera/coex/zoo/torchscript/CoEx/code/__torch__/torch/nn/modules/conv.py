class Conv2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Optional[Tensor]
  training : bool
  _is_full_backward_hook : None
  transposed : bool
  _reversed_padding_repeated_twice : Tuple[int, int, int, int]
  out_channels : Final[int] = 32
  kernel_size : Final[Tuple[int, int]] = (3, 3)
  in_channels : Final[int] = 3
  output_padding : Final[Tuple[int, int]] = (0, 0)
  dilation : Final[Tuple[int, int]] = (1, 1)
  stride : Final[Tuple[int, int]] = (2, 2)
  padding : Final[Tuple[int, int]] = (1, 1)
  groups : Final[int] = 1
  padding_mode : Final[str] = "zeros"
  def forward(self: __torch__.torch.nn.modules.conv.Conv2d,
    input: Tensor) -> Tensor:
    _0 = (self)._conv_forward(input, self.weight, self.bias, )
    return _0
  def _conv_forward(self: __torch__.torch.nn.modules.conv.Conv2d,
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor]) -> Tensor:
    _1 = torch.conv2d(input, weight, bias, [2, 2], [1, 1], [1, 1], 1)
    return _1
class ConvTranspose2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Optional[Tensor]
  training : bool
  _is_full_backward_hook : None
  transposed : bool
  _reversed_padding_repeated_twice : Tuple[int, int, int, int]
  out_channels : Final[int] = 96
  kernel_size : Final[Tuple[int, int]] = (4, 4)
  in_channels : Final[int] = 160
  output_padding : Final[Tuple[int, int]] = (0, 0)
  dilation : Final[Tuple[int, int]] = (1, 1)
  stride : Final[Tuple[int, int]] = (2, 2)
  padding : Final[Tuple[int, int]] = (1, 1)
  groups : Final[int] = 1
  padding_mode : Final[str] = "zeros"
  def forward(self: __torch__.torch.nn.modules.conv.ConvTranspose2d,
    input: Tensor,
    output_size: Optional[List[int]]=None) -> Tensor:
    output_padding = (self)._output_padding(input, output_size, [2, 2], [1, 1], [4, 4], [1, 1], )
    _2 = torch.conv_transpose2d(input, self.weight, self.bias, [2, 2], [1, 1], output_padding, 1, [1, 1])
    return _2
  def _output_padding(self: __torch__.torch.nn.modules.conv.ConvTranspose2d,
    input: Tensor,
    output_size: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    kernel_size: List[int],
    dilation: Optional[List[int]]=None) -> List[int]:
    _3 = "output_size must have {} or {} elements (got {})"
    _4 = "requested an output size of {}, but valid sizes range from {} to {} (for an input of {})"
    if torch.__is__(output_size, None):
      ret = [0, 0]
    else:
      output_size0 = unchecked_cast(List[int], output_size)
      k = torch.sub(torch.dim(input), 2)
      _5 = torch.eq(torch.len(output_size0), torch.add(k, 2))
      if _5:
        output_size2 = torch.slice(output_size0, 2, 9223372036854775807, 1)
        output_size1 = output_size2
      else:
        output_size1 = output_size0
      _6 = torch.ne(torch.len(output_size1), k)
      if _6:
        _7 = torch.format(_3, k, torch.add(k, 2), torch.len(output_size1))
        ops.prim.RaiseException(_7)
      else:
        pass
      min_sizes = annotate(List[int], [])
      max_sizes = annotate(List[int], [])
      dilation0 = dilation
      for d in range(k):
        _8 = torch.size(input, torch.add(d, 2))
        _9 = torch.mul(torch.sub(_8, 1), stride[d])
        _10 = torch.sub(_9, torch.mul(2, padding[d]))
        _11 = torch.__isnot__(dilation0, None)
        if _11:
          dilation2 = unchecked_cast(List[int], dilation0)
          _12, dilation1 = dilation2[d], dilation2
        else:
          _12, dilation1 = 1, dilation0
        _13 = torch.mul(_12, torch.sub(kernel_size[d], 1))
        dim_size = torch.add(torch.add(_10, _13), 1)
        _14 = torch.append(min_sizes, dim_size)
        _15 = torch.add(min_sizes[d], stride[d])
        _16 = torch.append(max_sizes, torch.sub(_15, 1))
        dilation0 = dilation1
      for i in range(torch.len(output_size1)):
        size = output_size1[i]
        min_size = min_sizes[i]
        max_size = max_sizes[i]
        if torch.lt(size, min_size):
          _17 = True
        else:
          _17 = torch.gt(size, max_size)
        if _17:
          _18 = torch.slice(torch.size(input), 2, 9223372036854775807, 1)
          _19 = torch.format(_4, output_size1, min_sizes, max_sizes, _18)
          ops.prim.RaiseException(_19)
        else:
          pass
      res = annotate(List[int], [])
      for d0 in range(k):
        _20 = torch.sub(output_size1[d0], min_sizes[d0])
        _21 = torch.append(res, _20)
      ret = res
    return ret
class Conv3d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Optional[Tensor]
  training : bool
  _is_full_backward_hook : None
  transposed : bool
  _reversed_padding_repeated_twice : Tuple[int, int, int, int, int, int]
  out_channels : Final[int] = 8
  kernel_size : Final[Tuple[int, int, int]] = (3, 3, 3)
  in_channels : Final[int] = 1
  output_padding : Final[Tuple[int, int, int]] = (0, 0, 0)
  dilation : Final[Tuple[int, int, int]] = (1, 1, 1)
  stride : Final[Tuple[int, int, int]] = (1, 1, 1)
  padding : Final[Tuple[int, int, int]] = (1, 1, 1)
  groups : Final[int] = 1
  padding_mode : Final[str] = "zeros"
  def forward(self: __torch__.torch.nn.modules.conv.Conv3d,
    input: Tensor) -> Tensor:
    _22 = torch.conv3d(input, self.weight, self.bias, [1, 1, 1], [1, 1, 1], [1, 1, 1], 1)
    return _22
class ConvTranspose3d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Optional[Tensor]
  training : bool
  _is_full_backward_hook : None
  transposed : bool
  _reversed_padding_repeated_twice : Tuple[int, int, int, int, int, int]
  out_channels : Final[int] = 1
  kernel_size : Final[Tuple[int, int, int]] = (4, 4, 4)
  in_channels : Final[int] = 16
  output_padding : Final[Tuple[int, int, int]] = (0, 0, 0)
  dilation : Final[Tuple[int, int, int]] = (1, 1, 1)
  stride : Final[Tuple[int, int, int]] = (2, 2, 2)
  padding : Final[Tuple[int, int, int]] = (1, 1, 1)
  groups : Final[int] = 1
  padding_mode : Final[str] = "zeros"
  def forward(self: __torch__.torch.nn.modules.conv.ConvTranspose3d,
    input: Tensor,
    output_size: Optional[List[int]]=None) -> Tensor:
    output_padding = (self)._output_padding(input, output_size, [2, 2, 2], [1, 1, 1], [4, 4, 4], [1, 1, 1], )
    _23 = torch.conv_transpose3d(input, self.weight, self.bias, [2, 2, 2], [1, 1, 1], output_padding, 1, [1, 1, 1])
    return _23
  def _output_padding(self: __torch__.torch.nn.modules.conv.ConvTranspose3d,
    input: Tensor,
    output_size: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    kernel_size: List[int],
    dilation: Optional[List[int]]=None) -> List[int]:
    _24 = "output_size must have {} or {} elements (got {})"
    _25 = "requested an output size of {}, but valid sizes range from {} to {} (for an input of {})"
    if torch.__is__(output_size, None):
      ret = [0, 0, 0]
    else:
      output_size3 = unchecked_cast(List[int], output_size)
      k = torch.sub(torch.dim(input), 2)
      _26 = torch.eq(torch.len(output_size3), torch.add(k, 2))
      if _26:
        output_size5 = torch.slice(output_size3, 2, 9223372036854775807, 1)
        output_size4 = output_size5
      else:
        output_size4 = output_size3
      _27 = torch.ne(torch.len(output_size4), k)
      if _27:
        _28 = torch.format(_24, k, torch.add(k, 2), torch.len(output_size4))
        ops.prim.RaiseException(_28)
      else:
        pass
      min_sizes = annotate(List[int], [])
      max_sizes = annotate(List[int], [])
      dilation3 = dilation
      for d in range(k):
        _29 = torch.size(input, torch.add(d, 2))
        _30 = torch.mul(torch.sub(_29, 1), stride[d])
        _31 = torch.sub(_30, torch.mul(2, padding[d]))
        _32 = torch.__isnot__(dilation3, None)
        if _32:
          dilation5 = unchecked_cast(List[int], dilation3)
          _33, dilation4 = dilation5[d], dilation5
        else:
          _33, dilation4 = 1, dilation3
        _34 = torch.mul(_33, torch.sub(kernel_size[d], 1))
        dim_size = torch.add(torch.add(_31, _34), 1)
        _35 = torch.append(min_sizes, dim_size)
        _36 = torch.add(min_sizes[d], stride[d])
        _37 = torch.append(max_sizes, torch.sub(_36, 1))
        dilation3 = dilation4
      for i in range(torch.len(output_size4)):
        size = output_size4[i]
        min_size = min_sizes[i]
        max_size = max_sizes[i]
        if torch.lt(size, min_size):
          _38 = True
        else:
          _38 = torch.gt(size, max_size)
        if _38:
          _39 = torch.slice(torch.size(input), 2, 9223372036854775807, 1)
          _40 = torch.format(_25, output_size4, min_sizes, max_sizes, _39)
          ops.prim.RaiseException(_40)
        else:
          pass
      res = annotate(List[int], [])
      for d1 in range(k):
        _41 = torch.sub(output_size4[d1], min_sizes[d1])
        _42 = torch.append(res, _41)
      ret = res
    return ret
