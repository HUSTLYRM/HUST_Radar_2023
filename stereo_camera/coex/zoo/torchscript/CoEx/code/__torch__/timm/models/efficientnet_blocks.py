class DepthwiseSeparableConv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  has_residual : bool
  has_pw_act : bool
  drop_path_rate : float
  se : None
  conv_dw : __torch__.torch.nn.modules.conv.___torch_mangle_0.Conv2d
  bn1 : __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  act1 : __torch__.torch.nn.modules.activation.ReLU6
  conv_pw : __torch__.torch.nn.modules.conv.___torch_mangle_1.Conv2d
  bn2 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_2.BatchNorm2d
  act2 : __torch__.torch.nn.modules.linear.Identity
  def forward(self: __torch__.timm.models.efficientnet_blocks.DepthwiseSeparableConv,
    x: Tensor) -> Tensor:
    _0 = __torch__.timm.models.layers.drop.drop_path
    x0 = (self.conv_dw).forward(x, )
    x1 = (self.bn1).forward(x0, )
    x2 = (self.act1).forward(x1, )
    x3 = (self.conv_pw).forward(x2, )
    x4 = (self.bn2).forward(x3, )
    x5 = (self.act2).forward(x4, )
    if self.has_residual:
      if torch.gt(self.drop_path_rate, 0.):
        x8 = _0(x5, self.drop_path_rate, self.training, )
        x7 = x8
      else:
        x7 = x5
      x6 = torch.add_(x7, x, alpha=1)
    else:
      x6 = x5
    return x6
class InvertedResidual(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  has_residual : bool
  drop_path_rate : float
  se : None
  conv_pw : __torch__.torch.nn.modules.conv.___torch_mangle_4.Conv2d
  bn1 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_5.BatchNorm2d
  act1 : __torch__.torch.nn.modules.activation.ReLU6
  conv_dw : __torch__.torch.nn.modules.conv.___torch_mangle_6.Conv2d
  bn2 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_5.BatchNorm2d
  act2 : __torch__.torch.nn.modules.activation.ReLU6
  conv_pwl : __torch__.torch.nn.modules.conv.___torch_mangle_7.Conv2d
  bn3 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_8.BatchNorm2d
  def forward(self: __torch__.timm.models.efficientnet_blocks.InvertedResidual,
    x: Tensor) -> Tensor:
    _1 = __torch__.timm.models.layers.drop.drop_path
    x9 = (self.conv_pw).forward(x, )
    x10 = (self.bn1).forward(x9, )
    x11 = (self.act1).forward(x10, )
    x12 = (self.conv_dw).forward(x11, )
    x13 = (self.bn2).forward(x12, )
    x14 = (self.act2).forward(x13, )
    x15 = (self.conv_pwl).forward(x14, )
    x16 = (self.bn3).forward(x15, )
    if self.has_residual:
      if torch.gt(self.drop_path_rate, 0.):
        x19 = _1(x16, self.drop_path_rate, self.training, )
        x18 = x19
      else:
        x18 = x16
      x17 = torch.add_(x18, x, alpha=1)
    else:
      x17 = x16
    return x17
