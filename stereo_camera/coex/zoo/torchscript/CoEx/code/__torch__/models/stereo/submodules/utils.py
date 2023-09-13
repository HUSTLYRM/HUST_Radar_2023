class AttentionCostVolume(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  head : int
  weighted : bool
  costVolume : __torch__.models.stereo.submodules.utils.CostVolume
  conv : __torch__.models.stereo.submodules.util_conv.___torch_mangle_70.BasicConv
  desc : __torch__.torch.nn.modules.conv.___torch_mangle_71.Conv2d
  def forward(self: __torch__.models.stereo.submodules.utils.AttentionCostVolume,
    imL: Tensor,
    imR: Tensor) -> Tensor:
    _0 = __torch__.torch.functional.___torch_mangle_153.norm
    x = (self.conv).forward(imL, )
    y = (self.conv).forward(imR, )
    x_ = (self.desc).forward(x, )
    y_ = (self.desc).forward(y, )
    _1 = self.costVolume
    _2 = torch.div(x_, _0(x_, 2, 1, True, None, None, ))
    _3 = torch.div(y_, _0(y_, 2, 1, True, None, None, ))
    return (_1).forward(_2, _3, )
class CostVolume(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  maxdisp : int
  glue : bool
  group : int
  unfold : __torch__.torch.nn.modules.fold.Unfold
  left_pad : __torch__.torch.nn.modules.padding.ZeroPad2d
  def forward(self: __torch__.models.stereo.submodules.utils.CostVolume,
    x: Tensor,
    y: Tensor) -> Tensor:
    b, c, h, w, = torch.size(x)
    _4 = (self.unfold).forward((self.left_pad).forward(y, ), )
    _5 = [b, self.group, torch.floordiv(c, self.group), self.maxdisp, h, w]
    unfolded_y = torch.reshape(_4, _5)
    _6 = [b, self.group, torch.floordiv(c, self.group), 1, h, w]
    x0 = torch.reshape(x, _6)
    cost = torch.sum(torch.mul(x0, unfolded_y), [2], False, dtype=None)
    return torch.flip(cost, [2])
class channelAtt(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  im_att : __torch__.torch.nn.modules.container.___torch_mangle_76.Sequential
  def forward(self: __torch__.models.stereo.submodules.utils.channelAtt,
    cv: Tensor,
    im: Tensor) -> Tensor:
    channel_att = torch.unsqueeze((self.im_att).forward(im, ), 2)
    cv0 = torch.mul(torch.sigmoid(channel_att), cv)
    return cv0
