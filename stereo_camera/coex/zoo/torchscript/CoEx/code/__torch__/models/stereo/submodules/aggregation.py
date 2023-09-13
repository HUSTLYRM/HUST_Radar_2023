class Aggregation(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  D : int
  gce : bool
  conv_stem : __torch__.models.stereo.submodules.util_conv.___torch_mangle_72.BasicConv
  channelAttStem : __torch__.models.stereo.submodules.utils.channelAtt
  channelAtt : __torch__.torch.nn.modules.container.ModuleList
  channelAttDown : __torch__.torch.nn.modules.container.___torch_mangle_97.ModuleList
  conv_down : __torch__.torch.nn.modules.container.___torch_mangle_116.ModuleList
  conv_up : __torch__.torch.nn.modules.container.___torch_mangle_123.ModuleList
  conv_skip : __torch__.torch.nn.modules.container.___torch_mangle_130.ModuleList
  conv_agg : __torch__.torch.nn.modules.container.___torch_mangle_136.ModuleList
  def forward(self: __torch__.models.stereo.submodules.aggregation.Aggregation,
    x4: Tensor,
    x8: Tensor,
    x16: Tensor,
    x32: Tensor,
    cost: Tensor) -> Tensor:
    b, c, h, w, = torch.size(x4)
    cost0 = torch.reshape(cost, [b, -1, self.D, h, w])
    cost1 = (self.conv_stem).forward(cost0, )
    cost2 = (self.channelAttStem).forward(cost1, x4, )
    cost_feat = [cost2]
    cost_ = (getattr(self.conv_down, "0")).forward(cost2, )
    cost_0 = (getattr(self.channelAttDown, "0")).forward(cost_, x8, )
    _0 = torch.append(cost_feat, cost_0)
    cost_1 = (getattr(self.conv_down, "1")).forward(cost_0, )
    cost_2 = (getattr(self.channelAttDown, "1")).forward(cost_1, x16, )
    _1 = torch.append(cost_feat, cost_2)
    cost_3 = (getattr(self.conv_down, "2")).forward(cost_2, )
    cost_4 = (getattr(self.channelAttDown, "2")).forward(cost_3, x32, )
    _2 = torch.append(cost_feat, cost_4)
    cost_5 = cost_feat[-1]
    cost_6 = (getattr(self.conv_up, "2")).forward(cost_5, )
    cost_7 = torch.cat([cost_6, cost_feat[-2]], 1)
    cost_8 = (getattr(self.conv_skip, "2")).forward(cost_7, )
    cost_9 = (getattr(self.conv_agg, "2")).forward(cost_8, )
    cost_10 = (getattr(self.channelAtt, "2")).forward(cost_9, x16, )
    cost_11 = (getattr(self.conv_up, "1")).forward(cost_10, )
    cost_12 = torch.cat([cost_11, cost_feat[-3]], 1)
    cost_13 = (getattr(self.conv_skip, "1")).forward(cost_12, )
    cost_14 = (getattr(self.conv_agg, "1")).forward(cost_13, )
    cost_15 = (getattr(self.channelAtt, "1")).forward(cost_14, x8, )
    cost_16 = (getattr(self.conv_up, "0")).forward(cost_15, )
    return cost_16
