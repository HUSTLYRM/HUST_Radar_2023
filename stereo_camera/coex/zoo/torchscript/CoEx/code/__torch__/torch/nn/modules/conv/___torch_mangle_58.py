class ConvTranspose2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Optional[Tensor]
  training : bool
  _is_full_backward_hook : None
  transposed : bool
  _reversed_padding_repeated_twice : Tuple[int, int, int, int]
  out_channels : Final[int] = 32
  kernel_size : Final[Tuple[int, int]] = (4, 4)
  in_channels : Final[int] = 192
  output_padding : Final[Tuple[int, int]] = (0, 0)
  dilation : Final[Tuple[int, int]] = (1, 1)
  stride : Final[Tuple[int, int]] = (2, 2)
  padding : Final[Tuple[int, int]] = (1, 1)
  groups : Final[int] = 1
  padding_mode : Final[str] = "zeros"
  def forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_58.ConvTranspose2d,
    input: Tensor,
    output_size: Optional[List[int]]=None) -> Tensor:
    output_padding = (self)._output_padding(input, output_size, [2, 2], [1, 1], [4, 4], [1, 1], )
    _0 = torch.conv_transpose2d(input, self.weight, self.bias, [2, 2], [1, 1], output_padding, 1, [1, 1])
    return _0
  def _output_padding(self: __torch__.torch.nn.modules.conv.___torch_mangle_58.ConvTranspose2d,
    input: Tensor,
    output_size: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    kernel_size: List[int],
    dilation: Optional[List[int]]=None) -> List[int]:
    _1 = "output_size must have {} or {} elements (got {})"
    _2 = "requested an output size of {}, but valid sizes range from {} to {} (for an input of {})"
    if torch.__is__(output_size, None):
      ret = [0, 0]
    else:
      output_size0 = unchecked_cast(List[int], output_size)
      k = torch.sub(torch.dim(input), 2)
      _3 = torch.eq(torch.len(output_size0), torch.add(k, 2))
      if _3:
        output_size2 = torch.slice(output_size0, 2, 9223372036854775807, 1)
        output_size1 = output_size2
      else:
        output_size1 = output_size0
      _4 = torch.ne(torch.len(output_size1), k)
      if _4:
        _5 = torch.format(_1, k, torch.add(k, 2), torch.len(output_size1))
        ops.prim.RaiseException(_5)
      else:
        pass
      min_sizes = annotate(List[int], [])
      max_sizes = annotate(List[int], [])
      dilation0 = dilation
      for d in range(k):
        _6 = torch.size(input, torch.add(d, 2))
        _7 = torch.mul(torch.sub(_6, 1), stride[d])
        _8 = torch.sub(_7, torch.mul(2, padding[d]))
        _9 = torch.__isnot__(dilation0, None)
        if _9:
          dilation2 = unchecked_cast(List[int], dilation0)
          _10, dilation1 = dilation2[d], dilation2
        else:
          _10, dilation1 = 1, dilation0
        _11 = torch.mul(_10, torch.sub(kernel_size[d], 1))
        dim_size = torch.add(torch.add(_8, _11), 1)
        _12 = torch.append(min_sizes, dim_size)
        _13 = torch.add(min_sizes[d], stride[d])
        _14 = torch.append(max_sizes, torch.sub(_13, 1))
        dilation0 = dilation1
      for i in range(torch.len(output_size1)):
        size = output_size1[i]
        min_size = min_sizes[i]
        max_size = max_sizes[i]
        if torch.lt(size, min_size):
          _15 = True
        else:
          _15 = torch.gt(size, max_size)
        if _15:
          _16 = torch.slice(torch.size(input), 2, 9223372036854775807, 1)
          _17 = torch.format(_2, output_size1, min_sizes, max_sizes, _16)
          ops.prim.RaiseException(_17)
        else:
          pass
      res = annotate(List[int], [])
      for d0 in range(k):
        _18 = torch.sub(output_size1[d0], min_sizes[d0])
        _19 = torch.append(res, _18)
      ret = res
    return ret
