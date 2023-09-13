class CoEx(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  type : str
  D : int
  corr_volume : bool
  feature : __torch__.models.stereo.submodules.feature.Feature
  up : __torch__.models.stereo.submodules.feature.FeatUp
  cost_volume : __torch__.models.stereo.submodules.utils.AttentionCostVolume
  cost_agg : __torch__.models.stereo.submodules.aggregation.Aggregation
  regression : __torch__.models.stereo.submodules.regression.Regression
  stem_2 : __torch__.torch.nn.modules.container.___torch_mangle_139.Sequential
  stem_4 : __torch__.torch.nn.modules.container.___torch_mangle_142.Sequential
  spx : __torch__.torch.nn.modules.container.___torch_mangle_144.Sequential
  spx_2 : __torch__.models.stereo.submodules.util_conv.___torch_mangle_147.Conv2x
  spx_4 : __torch__.torch.nn.modules.container.___torch_mangle_151.Sequential
  def forward(self: __torch__.models.stereo.CoEx.CoEx,
    imL: Tensor,
    imR: Tensor) -> List[Tensor]:
    _0 = __torch__.torch.nn.functional.softmax
    _1 = torch.eq(torch.size(imL), torch.size(imR))
    if _1:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    x2, x, = (self.feature).forward(imL, )
    y2, y, = (self.feature).forward(imR, )
    _2 = (self.up).forward(x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], )
    x0, y0, = _2
    stem_2x = (self.stem_2).forward(imL, )
    stem_4x = (self.stem_4).forward(stem_2x, )
    stem_2y = (self.stem_2).forward(imR, )
    stem_4y = (self.stem_4).forward(stem_2y, )
    _3 = torch._set_item(x0, 0, torch.cat([x0[0], stem_4x], 1))
    _4 = torch._set_item(y0, 0, torch.cat([y0[0], stem_4y], 1))
    _5 = (self.cost_volume).forward(x0[0], y0[0], )
    _6 = torch.slice(_5, 0, 0, 9223372036854775807, 1)
    _7 = torch.slice(_6, 1, 0, 9223372036854775807, 1)
    cost = torch.slice(_7, 2, 0, -1, 1)
    cost0 = (self.cost_agg).forward(x0[0], x0[1], x0[2], x0[3], cost, )
    xspx = (self.spx_4).forward(x0[0], )
    xspx0 = (self.spx_2).forward(xspx, stem_2x, )
    spx_pred = (self.spx).forward(xspx0, )
    spx_pred0 = _0(spx_pred, 1, 3, None, )
    disp_pred = (self.regression).forward(cost0, spx_pred0, )
    return disp_pred
