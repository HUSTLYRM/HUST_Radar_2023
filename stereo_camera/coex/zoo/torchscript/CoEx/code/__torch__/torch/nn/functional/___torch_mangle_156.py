def interpolate(input: Tensor,
    size: Optional[int]=None,
    scale_factor: Optional[float]=None,
    mode: str="nearest",
    align_corners: Optional[bool]=None,
    recompute_scale_factor: Optional[bool]=None) -> Tensor:
  _0 = "align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear"
  _1 = "Default upsampling behavior when mode={} is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details."
  _2 = "only one of size or scale_factor should be defined"
  _3 = "either size or scale_factor should be defined"
  _4 = "The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0 to align with other frameworks/libraries, and now uses scale_factor directly, instead of relying on the computed output size. If you wish to restore the old behavior, please set recompute_scale_factor=True. See the documentation of nn.Upsample for details. "
  _5 = "recompute_scale_factor is not meaningful with an explicit size."
  _6 = __torch__.torch.nn.functional.adaptive_avg_pool2d
  _7 = __torch__.torch.nn.functional.adaptive_avg_pool3d
  _8 = "Got 3D input, but bilinear mode needs 4D input"
  _9 = "Got 3D input, but trilinear mode needs 5D input"
  _10 = "Got 4D input, but linear mode needs 3D input"
  _11 = "Got 4D input, but trilinear mode needs 5D input"
  _12 = "Got 5D input, but linear mode needs 3D input"
  _13 = "Got 5D input, but bilinear mode needs 4D input"
  _14 = "Input Error: Only 3D, 4D and 5D input Tensors supported (got {}D) for the modes: nearest | linear | bilinear | bicubic | trilinear (got {})"
  _15 = uninitialized(Tensor)
  _16 = uninitialized(List[int])
  _17 = uninitialized(None)
  _18 = uninitialized(List[float])
  _19 = uninitialized(Optional[List[int]])
  _20 = uninitialized(Optional[List[float]])
  _21 = uninitialized(Optional[int])
  _22 = uninitialized(Optional[bool])
  _23 = uninitialized(bool)
  _24 = torch.__contains__(["nearest", "area"], mode)
  if _24:
    _25 = torch.__isnot__(align_corners, None)
    if _25:
      ops.prim.RaiseException(_0)
      align_corners1 = _22
    else:
      align_corners1 = align_corners
    align_corners0 = align_corners1
  else:
    if torch.__is__(align_corners, None):
      torch.warn(torch.format(_1, mode), 2)
      align_corners2 = False
    else:
      align_corners3 = unchecked_cast(bool, align_corners)
      align_corners2 = align_corners3
    align_corners0 = align_corners2
  dim = torch.sub(torch.dim(input), 2)
  if torch.__isnot__(size, None):
    size1 = unchecked_cast(int, size)
    _26, size0 = torch.__isnot__(scale_factor, None), size1
  else:
    _26, size0 = False, size
  if _26:
    ops.prim.RaiseException(_2)
    scale_factors, size2, output_size = _20, _21, _19
  else:
    if torch.__isnot__(size0, None):
      size4 = unchecked_cast(int, size0)
      if torch.__is__(scale_factor, None):
        pass
      else:
        ops.prim.RaiseException("AssertionError: ")
      output_size1 = annotate(List[int], [])
      for _27 in range(dim):
        _28 = torch.append(output_size1, size4)
      scale_factors0, size3, output_size0 = None, size4, output_size1
    else:
      _29 = torch.__isnot__(scale_factor, None)
      if _29:
        scale_factor0 = unchecked_cast(float, scale_factor)
        if torch.__is__(size0, None):
          size6 = size0
        else:
          ops.prim.RaiseException("AssertionError: ")
          size6 = _21
        scale_factors2 = annotate(List[float], [])
        for _30 in range(dim):
          _31 = torch.append(scale_factors2, scale_factor0)
        scale_factors1, size5, output_size2 = scale_factors2, size6, None
      else:
        ops.prim.RaiseException(_3)
        scale_factors1, size5, output_size2 = _18, _21, _17
      scale_factors0, size3, output_size0 = scale_factors1, size5, output_size2
    scale_factors, size2, output_size = scale_factors0, size3, output_size0
  _32 = torch.__is__(recompute_scale_factor, None)
  if _32:
    _33 = torch.__isnot__(scale_factors, None)
    if _33:
      scale_factors5 = unchecked_cast(List[float], scale_factors)
      _34 = torch.len(scale_factors5)
      _35 = 0
      _36 = torch.gt(_34, 0)
      while _36:
        scale = scale_factors5[_35]
        _37 = torch.ne(torch.floor(scale), scale)
        if _37:
          torch.warn(_4, 2)
          _38 = False
        else:
          _38 = True
        _39 = torch.add(_35, 1)
        _40 = torch.__and__(torch.lt(_39, _34), _38)
        _36, _35 = _40, _39
      scale_factors4 = scale_factors5
    else:
      scale_factors4 = scale_factors
    recompute_scale_factor0, scale_factors3 = recompute_scale_factor, scale_factors4
  else:
    recompute_scale_factor1 = unchecked_cast(bool, recompute_scale_factor)
    if recompute_scale_factor1:
      _41 = torch.__isnot__(size2, None)
    else:
      _41 = False
    if _41:
      ops.prim.RaiseException(_5)
    else:
      pass
    recompute_scale_factor0, scale_factors3 = recompute_scale_factor1, scale_factors
  if torch.eq(mode, "area"):
    _42 = torch.__is__(output_size, None)
  else:
    _42 = False
  if _42:
    recompute_scale_factor2 = True
  else:
    recompute_scale_factor2 = recompute_scale_factor0
  _43 = torch.__isnot__(recompute_scale_factor2, None)
  if _43:
    recompute_scale_factor3 = unchecked_cast(bool, recompute_scale_factor2)
    _44 = recompute_scale_factor3
  else:
    _44 = False
  if _44:
    _45 = torch.__isnot__(scale_factors3, None)
    if _45:
      scale_factors8 = unchecked_cast(List[float], scale_factors3)
      scale_factors7 = scale_factors8
    else:
      ops.prim.RaiseException("AssertionError: ")
      scale_factors7 = _18
    output_size4 = annotate(List[int], [])
    for i in range(dim):
      _46 = torch.size(input, torch.add(i, 2))
      _47 = torch.mul(float(_46), scale_factors7[i])
      _48 = torch.append(output_size4, torch.floor(_47))
    output_size3, scale_factors6 = output_size4, None
  else:
    output_size3, scale_factors6 = output_size, scale_factors3
  if torch.eq(torch.dim(input), 3):
    _49 = torch.eq(mode, "nearest")
  else:
    _49 = False
  if _49:
    _51 = torch.upsample_nearest1d(input, output_size3, scale_factors6)
    _50 = _51
  else:
    if torch.eq(torch.dim(input), 4):
      _52 = torch.eq(mode, "nearest")
    else:
      _52 = False
    if _52:
      _54 = torch.upsample_nearest2d(input, output_size3, scale_factors6)
      _53 = _54
    else:
      if torch.eq(torch.dim(input), 5):
        _55 = torch.eq(mode, "nearest")
      else:
        _55 = False
      if _55:
        _57 = torch.upsample_nearest3d(input, output_size3, scale_factors6)
        _56 = _57
      else:
        if torch.eq(torch.dim(input), 3):
          _58 = torch.eq(mode, "area")
        else:
          _58 = False
        if _58:
          _60 = torch.__isnot__(output_size3, None)
          if _60:
            output_size6 = unchecked_cast(List[int], output_size3)
            output_size5 = output_size6
          else:
            ops.prim.RaiseException("AssertionError: ")
            output_size5 = _16
          _61 = torch.adaptive_avg_pool1d(input, output_size5)
          _59 = _61
        else:
          if torch.eq(torch.dim(input), 4):
            _62 = torch.eq(mode, "area")
          else:
            _62 = False
          if _62:
            _64 = torch.__isnot__(output_size3, None)
            if _64:
              output_size8 = unchecked_cast(List[int], output_size3)
              output_size7 = output_size8
            else:
              ops.prim.RaiseException("AssertionError: ")
              output_size7 = _16
            _63 = _6(input, output_size7, )
          else:
            _65 = torch.eq(torch.dim(input), 5)
            if _65:
              _66 = torch.eq(mode, "area")
            else:
              _66 = False
            if _66:
              _68 = torch.__isnot__(output_size3, None)
              if _68:
                output_size10 = unchecked_cast(List[int], output_size3)
                output_size9 = output_size10
              else:
                ops.prim.RaiseException("AssertionError: ")
                output_size9 = _16
              _67 = _7(input, output_size9, )
            else:
              _69 = torch.eq(torch.dim(input), 3)
              if _69:
                _71 = torch.eq(mode, "linear")
                _70 = _71
              else:
                _70 = False
              if _70:
                _73 = torch.__isnot__(align_corners0, None)
                if _73:
                  align_corners5 = unchecked_cast(bool, align_corners0)
                  align_corners4 = align_corners5
                else:
                  ops.prim.RaiseException("AssertionError: ")
                  align_corners4 = _23
                _74 = torch.upsample_linear1d(input, output_size3, align_corners4, scale_factors6)
                _72 = _74
              else:
                _75 = torch.eq(torch.dim(input), 4)
                if _75:
                  _77 = torch.eq(mode, "bilinear")
                  _76 = _77
                else:
                  _76 = False
                if _76:
                  _79 = torch.__isnot__(align_corners0, None)
                  if _79:
                    align_corners7 = unchecked_cast(bool, align_corners0)
                    align_corners6 = align_corners7
                  else:
                    ops.prim.RaiseException("AssertionError: ")
                    align_corners6 = _23
                  _80 = torch.upsample_bilinear2d(input, output_size3, align_corners6, scale_factors6)
                  _78 = _80
                else:
                  _81 = torch.eq(torch.dim(input), 5)
                  if _81:
                    _83 = torch.eq(mode, "trilinear")
                    _82 = _83
                  else:
                    _82 = False
                  if _82:
                    _85 = torch.__isnot__(align_corners0, None)
                    if _85:
                      align_corners9 = unchecked_cast(bool, align_corners0)
                      align_corners8 = align_corners9
                    else:
                      ops.prim.RaiseException("AssertionError: ")
                      align_corners8 = _23
                    _86 = torch.upsample_trilinear3d(input, output_size3, align_corners8, scale_factors6)
                    _84 = _86
                  else:
                    _87 = torch.eq(torch.dim(input), 4)
                    if _87:
                      _89 = torch.eq(mode, "bicubic")
                      _88 = _89
                    else:
                      _88 = False
                    if _88:
                      _91 = torch.__isnot__(align_corners0, None)
                      if _91:
                        align_corners11 = unchecked_cast(bool, align_corners0)
                        align_corners10 = align_corners11
                      else:
                        ops.prim.RaiseException("AssertionError: ")
                        align_corners10 = _23
                      _92 = torch.upsample_bicubic2d(input, output_size3, align_corners10, scale_factors6)
                      _90 = _92
                    else:
                      _93 = torch.eq(torch.dim(input), 3)
                      if _93:
                        _95 = torch.eq(mode, "bilinear")
                        _94 = _95
                      else:
                        _94 = False
                      if _94:
                        ops.prim.RaiseException(_8)
                      else:
                        pass
                      _96 = torch.eq(torch.dim(input), 3)
                      if _96:
                        _98 = torch.eq(mode, "trilinear")
                        _97 = _98
                      else:
                        _97 = False
                      if _97:
                        ops.prim.RaiseException(_9)
                      else:
                        pass
                      _99 = torch.eq(torch.dim(input), 4)
                      if _99:
                        _101 = torch.eq(mode, "linear")
                        _100 = _101
                      else:
                        _100 = False
                      if _100:
                        ops.prim.RaiseException(_10)
                      else:
                        pass
                      _102 = torch.eq(torch.dim(input), 4)
                      if _102:
                        _104 = torch.eq(mode, "trilinear")
                        _103 = _104
                      else:
                        _103 = False
                      if _103:
                        ops.prim.RaiseException(_11)
                      else:
                        pass
                      _105 = torch.eq(torch.dim(input), 5)
                      if _105:
                        _107 = torch.eq(mode, "linear")
                        _106 = _107
                      else:
                        _106 = False
                      if _106:
                        ops.prim.RaiseException(_12)
                      else:
                        pass
                      _108 = torch.eq(torch.dim(input), 5)
                      if _108:
                        _110 = torch.eq(mode, "bilinear")
                        _109 = _110
                      else:
                        _109 = False
                      if _109:
                        ops.prim.RaiseException(_13)
                      else:
                        pass
                      _111 = torch.format(_14, torch.dim(input), mode)
                      ops.prim.RaiseException(_111)
                      _90 = _15
                    _84 = _90
                  _78 = _84
                _72 = _78
              _67 = _72
            _63 = _67
          _59 = _63
        _56 = _59
      _53 = _56
    _50 = _53
  return _50
