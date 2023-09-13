def _pad(input: Tensor,
    pad: List[int],
    mode: str="constant",
    value: float=0.) -> Tensor:
  _0 = "AssertionError: Padding length must be divisible by 2"
  _1 = "AssertionError: Padding length too large"
  _2 = "Padding mode \"{}\"\" doesn\'t take in value argument"
  _3 = "AssertionError: 3D tensors expect 2 values for padding"
  _4 = __torch__.torch.nn.functional._pad_circular
  _5 = "AssertionError: 4D tensors expect 4 values for padding"
  _6 = "AssertionError: 5D tensors expect 6 values for padding"
  _7 = "Only 3D, 4D, 5D padding with non-constant padding are supported for now"
  _8 = uninitialized(Tensor)
  _9 = torch.eq(torch.remainder(torch.len(pad), 2), 0)
  if _9:
    pass
  else:
    ops.prim.RaiseException(_0)
  _10 = torch.le(torch.floordiv(torch.len(pad), 2), torch.dim(input))
  if _10:
    pass
  else:
    ops.prim.RaiseException(_1)
  if torch.eq(mode, "constant"):
    _12 = torch.constant_pad_nd(input, pad, value)
    _11 = _12
  else:
    if torch.eq(value, 0):
      pass
    else:
      _13 = torch.add("AssertionError: ", torch.format(_2, mode))
      ops.prim.RaiseException(_13)
    if torch.eq(torch.dim(input), 3):
      if torch.eq(torch.len(pad), 2):
        pass
      else:
        ops.prim.RaiseException(_3)
      if torch.eq(mode, "reflect"):
        _16 = torch.reflection_pad1d(input, pad)
        _15 = _16
      else:
        if torch.eq(mode, "replicate"):
          _18 = torch.replication_pad1d(input, pad)
          _17 = _18
        else:
          if torch.eq(mode, "circular"):
            _19 = _4(input, pad, )
          else:
            ops.prim.RaiseException("")
            _19 = _8
          _17 = _19
        _15 = _17
      _14 = _15
    else:
      if torch.eq(torch.dim(input), 4):
        if torch.eq(torch.len(pad), 4):
          pass
        else:
          ops.prim.RaiseException(_5)
        if torch.eq(mode, "reflect"):
          _22 = torch.reflection_pad2d(input, pad)
          _21 = _22
        else:
          if torch.eq(mode, "replicate"):
            _24 = torch.replication_pad2d(input, pad)
            _23 = _24
          else:
            if torch.eq(mode, "circular"):
              _25 = _4(input, pad, )
            else:
              ops.prim.RaiseException("")
              _25 = _8
            _23 = _25
          _21 = _23
        _20 = _21
      else:
        if torch.eq(torch.dim(input), 5):
          if torch.eq(torch.len(pad), 6):
            pass
          else:
            ops.prim.RaiseException(_6)
          if torch.eq(mode, "reflect"):
            ops.prim.RaiseException("")
            _27 = _8
          else:
            if torch.eq(mode, "replicate"):
              _29 = torch.replication_pad3d(input, pad)
              _28 = _29
            else:
              _30 = torch.eq(mode, "circular")
              if _30:
                _31 = _4(input, pad, )
              else:
                ops.prim.RaiseException("")
                _31 = _8
              _28 = _31
            _27 = _28
          _26 = _27
        else:
          ops.prim.RaiseException(_7)
          _26 = _8
        _20 = _26
      _14 = _20
    _11 = _14
  return _11
def softmax(input: Tensor,
    dim: Optional[int]=None,
    _stacklevel: int=3,
    dtype: Optional[int]=None) -> Tensor:
  _32 = __torch__.torch.nn.functional._get_softmax_dim
  if torch.__is__(dim, None):
    dim1 = _32("softmax", torch.dim(input), _stacklevel, )
    dim0 = dim1
  else:
    dim0 = unchecked_cast(int, dim)
  if torch.__is__(dtype, None):
    ret = torch.softmax(input, dim0, None)
  else:
    dtype0 = unchecked_cast(int, dtype)
    ret = torch.softmax(input, dim0, dtype0)
  return ret
def batch_norm(input: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor]=None,
    bias: Optional[Tensor]=None,
    training: bool=False,
    momentum: float=0.10000000000000001,
    eps: float=1.0000000000000001e-05) -> Tensor:
  _33 = __torch__.torch.nn.functional._verify_batch_size
  if training:
    _34 = _33(torch.size(input), )
  else:
    pass
  _35 = torch.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, True)
  return _35
def hardtanh(input: Tensor,
    min_val: float=-1.,
    max_val: float=1.,
    inplace: bool=False) -> Tensor:
  if inplace:
    result0 = torch.hardtanh_(input, min_val, max_val)
    result = result0
  else:
    result1 = torch.hardtanh(input, min_val, max_val)
    result = result1
  return result
def leaky_relu(input: Tensor,
    negative_slope: float=0.01,
    inplace: bool=False) -> Tensor:
  if inplace:
    result2 = torch.leaky_relu_(input, negative_slope)
    result = result2
  else:
    result3 = torch.leaky_relu(input, negative_slope)
    result = result3
  return result
def unfold(input: Tensor,
    kernel_size: List[int],
    dilation: List[int]=[1, 1],
    padding: List[int]=[0, 0],
    stride: List[int]=[1, 1]) -> Tensor:
  _36 = "Input Error: Only 4D input Tensors are supported (got {}D)"
  _37 = uninitialized(Tensor)
  if torch.eq(torch.dim(input), 4):
    _39 = torch.im2col(input, kernel_size, dilation, padding, stride)
    _38 = _39
  else:
    ops.prim.RaiseException(torch.format(_36, torch.dim(input)))
    _38 = _37
  return _38
def relu(input: Tensor,
    inplace: bool=False) -> Tensor:
  if inplace:
    result = torch.relu_(input)
  else:
    result = torch.relu(input)
  return result
def _pad_circular(input: Tensor,
    padding: List[int]) -> Tensor:
  _40 = "AssertionError: Padding value causes wrapping around more than once."
  _41 = "AssertionError: Negative padding value is resulting in an empty dimension."
  in_shape = torch.size(input)
  paddable_shape = torch.slice(in_shape, 2, 9223372036854775807, 1)
  ndim = torch.len(paddable_shape)
  _42 = [9223372036854775807, torch.len(paddable_shape)]
  for idx in range(ops.prim.min(_42)):
    size = paddable_shape[idx]
    _43 = torch.neg(torch.add(torch.mul(idx, 2), 1))
    if torch.le(padding[_43], size):
      pass
    else:
      ops.prim.RaiseException(_40)
    _44 = torch.neg(torch.add(torch.mul(idx, 2), 2))
    if torch.le(padding[_44], size):
      pass
    else:
      ops.prim.RaiseException(_40)
    _45 = torch.neg(torch.add(torch.mul(idx, 2), 1))
    _46 = padding[_45]
    _47 = torch.neg(torch.add(torch.mul(idx, 2), 2))
    _48 = torch.add(torch.add(_46, padding[_47]), size)
    if torch.ge(_48, 0):
      pass
    else:
      ops.prim.RaiseException(_41)
  out_shape = torch.slice(in_shape, 0, 2, 1)
  _49 = [9223372036854775807, torch.len(paddable_shape)]
  out_shape0 = out_shape
  for idx0 in range(ops.prim.min(_49)):
    size0 = paddable_shape[idx0]
    _50 = torch.neg(torch.add(torch.mul(idx0, 2), 1))
    _51 = torch.add(size0, padding[_50])
    _52 = torch.neg(torch.add(torch.mul(idx0, 2), 2))
    out_shape1 = torch.add_(out_shape0, [torch.add(_51, padding[_52])])
    out_shape0 = out_shape1
  out = torch.empty(out_shape0, dtype=ops.prim.dtype(input), layout=ops.prim.layout(input), device=ops.prim.device(input), pin_memory=None, memory_format=None)
  if torch.eq(ndim, 1):
    out_d0 = ops.prim.max(padding[-2], 0)
    out_d1 = torch.sub(out_shape0[2], ops.prim.max(padding[-1], 0))
    in_d0 = ops.prim.max(torch.neg(padding[-2]), 0)
    _53 = in_shape[2]
    _54 = ops.prim.max(torch.neg(padding[-1]), 0)
    in_d1 = torch.sub(_53, _54)
    _55 = torch.slice(input, -1, in_d0, in_d1, 1)
    _56 = torch.slice(out, -1, out_d0, out_d1, 1)
    _57 = torch.copy_(_56, _55, False)
  else:
    if torch.eq(ndim, 2):
      out_d00 = ops.prim.max(padding[-2], 0)
      out_d10 = torch.sub(out_shape0[2], ops.prim.max(padding[-1], 0))
      out_h0 = ops.prim.max(padding[-4], 0)
      out_h1 = torch.sub(out_shape0[3], ops.prim.max(padding[-3], 0))
      in_d00 = ops.prim.max(torch.neg(padding[-2]), 0)
      _58 = in_shape[2]
      _59 = ops.prim.max(torch.neg(padding[-1]), 0)
      in_d10 = torch.sub(_58, _59)
      in_h0 = ops.prim.max(torch.neg(padding[-4]), 0)
      _60 = in_shape[3]
      _61 = ops.prim.max(torch.neg(padding[-3]), 0)
      in_h1 = torch.sub(_60, _61)
      _62 = torch.slice(input, -2, in_d00, in_d10, 1)
      _63 = torch.slice(_62, -1, in_h0, in_h1, 1)
      _64 = torch.slice(out, -2, out_d00, out_d10, 1)
      _65 = torch.slice(_64, -1, out_h0, out_h1, 1)
      _66 = torch.copy_(_65, _63, False)
    else:
      if torch.eq(ndim, 3):
        out_d01 = ops.prim.max(padding[-2], 0)
        out_d11 = torch.sub(out_shape0[2], ops.prim.max(padding[-1], 0))
        out_h00 = ops.prim.max(padding[-4], 0)
        out_h10 = torch.sub(out_shape0[3], ops.prim.max(padding[-3], 0))
        out_w0 = ops.prim.max(padding[-6], 0)
        out_w1 = torch.sub(out_shape0[4], ops.prim.max(padding[-5], 0))
        in_d01 = ops.prim.max(torch.neg(padding[-2]), 0)
        _67 = in_shape[2]
        _68 = ops.prim.max(torch.neg(padding[-1]), 0)
        in_d11 = torch.sub(_67, _68)
        in_h00 = ops.prim.max(torch.neg(padding[-4]), 0)
        _69 = in_shape[3]
        _70 = ops.prim.max(torch.neg(padding[-3]), 0)
        in_h10 = torch.sub(_69, _70)
        in_w0 = ops.prim.max(torch.neg(padding[-6]), 0)
        _71 = in_shape[4]
        _72 = ops.prim.max(torch.neg(padding[-5]), 0)
        in_w1 = torch.sub(_71, _72)
        _73 = torch.slice(input, -3, in_d01, in_d11, 1)
        _74 = torch.slice(_73, -2, in_h00, in_h10, 1)
        _75 = torch.slice(_74, -1, in_w0, in_w1, 1)
        _76 = torch.slice(out, -3, out_d01, out_d11, 1)
        _77 = torch.slice(_76, -2, out_h00, out_h10, 1)
        _78 = torch.slice(_77, -1, out_w0, out_w1, 1)
        _79 = torch.copy_(_78, _75, False)
      else:
        pass
  if torch.gt(padding[-2], 0):
    _80 = torch.sub(out_shape0[2], padding[-2])
    i0 = torch.sub(_80, ops.prim.max(padding[-1], 0))
    i1 = torch.sub(out_shape0[2], ops.prim.max(padding[-1], 0))
    o1 = padding[-2]
    _81 = torch.slice(out, 0, 0, 9223372036854775807, 1)
    _82 = torch.slice(_81, 1, 0, 9223372036854775807, 1)
    _83 = torch.slice(_82, 2, i0, i1, 1)
    _84 = torch.slice(out, 0, 0, 9223372036854775807, 1)
    _85 = torch.slice(_84, 1, 0, 9223372036854775807, 1)
    _86 = torch.copy_(torch.slice(_85, 2, 0, o1, 1), _83, False)
  else:
    pass
  if torch.gt(padding[-1], 0):
    i00 = ops.prim.max(padding[-2], 0)
    i10 = torch.add(ops.prim.max(padding[-2], 0), padding[-1])
    o0 = torch.sub(out_shape0[2], padding[-1])
    o10 = out_shape0[2]
    _87 = torch.slice(out, 0, 0, 9223372036854775807, 1)
    _88 = torch.slice(_87, 1, 0, 9223372036854775807, 1)
    _89 = torch.slice(_88, 2, i00, i10, 1)
    _90 = torch.slice(out, 0, 0, 9223372036854775807, 1)
    _91 = torch.slice(_90, 1, 0, 9223372036854775807, 1)
    _92 = torch.copy_(torch.slice(_91, 2, o0, o10, 1), _89, False)
  else:
    pass
  if torch.gt(torch.len(padding), 2):
    if torch.gt(padding[-4], 0):
      _93 = torch.sub(out_shape0[3], padding[-4])
      i01 = torch.sub(_93, ops.prim.max(padding[-3], 0))
      i11 = torch.sub(out_shape0[3], ops.prim.max(padding[-3], 0))
      o11 = padding[-4]
      _94 = torch.slice(out, 0, 0, 9223372036854775807, 1)
      _95 = torch.slice(_94, 1, 0, 9223372036854775807, 1)
      _96 = torch.slice(_95, 2, 0, 9223372036854775807, 1)
      _97 = torch.slice(_96, 3, i01, i11, 1)
      _98 = torch.slice(out, 0, 0, 9223372036854775807, 1)
      _99 = torch.slice(_98, 1, 0, 9223372036854775807, 1)
      _100 = torch.slice(_99, 2, 0, 9223372036854775807, 1)
      _101 = torch.copy_(torch.slice(_100, 3, 0, o11, 1), _97, False)
    else:
      pass
    if torch.gt(padding[-3], 0):
      i02 = ops.prim.max(padding[-4], 0)
      i12 = torch.add(ops.prim.max(padding[-4], 0), padding[-3])
      o00 = torch.sub(out_shape0[3], padding[-3])
      o12 = out_shape0[3]
      _102 = torch.slice(out, 0, 0, 9223372036854775807, 1)
      _103 = torch.slice(_102, 1, 0, 9223372036854775807, 1)
      _104 = torch.slice(_103, 2, 0, 9223372036854775807, 1)
      _105 = torch.slice(_104, 3, i02, i12, 1)
      _106 = torch.slice(out, 0, 0, 9223372036854775807, 1)
      _107 = torch.slice(_106, 1, 0, 9223372036854775807, 1)
      _108 = torch.slice(_107, 2, 0, 9223372036854775807, 1)
      _109 = torch.copy_(torch.slice(_108, 3, o00, o12, 1), _105, False)
    else:
      pass
  else:
    pass
  if torch.gt(torch.len(padding), 4):
    if torch.gt(padding[-6], 0):
      _110 = torch.sub(out_shape0[4], padding[-6])
      i03 = torch.sub(_110, ops.prim.max(padding[-5], 0))
      i13 = torch.sub(out_shape0[4], ops.prim.max(padding[-5], 0))
      o13 = padding[-6]
      _111 = torch.slice(out, 0, 0, 9223372036854775807, 1)
      _112 = torch.slice(_111, 1, 0, 9223372036854775807, 1)
      _113 = torch.slice(_112, 2, 0, 9223372036854775807, 1)
      _114 = torch.slice(_113, 3, 0, 9223372036854775807, 1)
      _115 = torch.slice(_114, 4, i03, i13, 1)
      _116 = torch.slice(out, 0, 0, 9223372036854775807, 1)
      _117 = torch.slice(_116, 1, 0, 9223372036854775807, 1)
      _118 = torch.slice(_117, 2, 0, 9223372036854775807, 1)
      _119 = torch.slice(_118, 3, 0, 9223372036854775807, 1)
      _120 = torch.copy_(torch.slice(_119, 4, 0, o13, 1), _115, False)
    else:
      pass
    if torch.gt(padding[-5], 0):
      i04 = ops.prim.max(padding[-6], 0)
      i14 = torch.add(ops.prim.max(padding[-6], 0), padding[-5])
      o01 = torch.sub(out_shape0[4], padding[-5])
      o14 = out_shape0[4]
      _121 = torch.slice(out, 0, 0, 9223372036854775807, 1)
      _122 = torch.slice(_121, 1, 0, 9223372036854775807, 1)
      _123 = torch.slice(_122, 2, 0, 9223372036854775807, 1)
      _124 = torch.slice(_123, 3, 0, 9223372036854775807, 1)
      _125 = torch.slice(_124, 4, i04, i14, 1)
      _126 = torch.slice(out, 0, 0, 9223372036854775807, 1)
      _127 = torch.slice(_126, 1, 0, 9223372036854775807, 1)
      _128 = torch.slice(_127, 2, 0, 9223372036854775807, 1)
      _129 = torch.slice(_128, 3, 0, 9223372036854775807, 1)
      _130 = torch.copy_(torch.slice(_129, 4, o01, o14, 1), _125, False)
    else:
      pass
  else:
    pass
  return out
def _get_softmax_dim(name: str,
    ndim: int,
    stacklevel: int) -> int:
  _131 = "Implicit dimension choice for {} has been deprecated. Change the call to include dim=X as an argument."
  torch.warn(torch.format(_131, name), stacklevel)
  if torch.eq(ndim, 0):
    _132 = True
  else:
    _132 = torch.eq(ndim, 1)
  if _132:
    _133 = True
  else:
    _133 = torch.eq(ndim, 3)
  if _133:
    ret = 0
  else:
    ret = 1
  return ret
def _verify_batch_size(size: List[int]) -> None:
  _134 = "Expected more than 1 value per channel when training, got input size {}"
  size_prods = size[0]
  size_prods0 = size_prods
  for i in range(torch.sub(torch.len(size), 2)):
    size_prods1 = torch.mul(size_prods0, size[torch.add(i, 2)])
    size_prods0 = size_prods1
  if torch.eq(size_prods0, 1):
    ops.prim.RaiseException(torch.format(_134, size))
  else:
    pass
  return None
def adaptive_avg_pool2d(input: Tensor,
    output_size: List[int]) -> Tensor:
  _135 = torch.gt(torch.len(torch.size(input)), torch.len(output_size))
  if _135:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _136 = torch.adaptive_avg_pool2d(input, output_size)
  return _136
def adaptive_avg_pool3d(input: Tensor,
    output_size: List[int]) -> Tensor:
  _137 = torch.gt(torch.len(torch.size(input)), torch.len(output_size))
  if _137:
    pass
  else:
    ops.prim.RaiseException("AssertionError: ")
  _138 = torch.adaptive_avg_pool3d(input, output_size)
  return _138
