class InvertedResidual(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  has_residual : bool
  drop_path_rate : float
  se : None
  conv_pw : __torch__.torch.nn.modules.conv.___torch_mangle_30.Conv2d
  bn1 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_31.BatchNorm2d
  act1 : __torch__.torch.nn.modules.activation.ReLU6
  conv_dw : __torch__.torch.nn.modules.conv.___torch_mangle_32.Conv2d
  bn2 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_31.BatchNorm2d
  act2 : __torch__.torch.nn.modules.activation.ReLU6
  conv_pwl : __torch__.torch.nn.modules.conv.___torch_mangle_36.Conv2d
  bn3 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_5.BatchNorm2d
  def forward(self: __torch__.timm.models.efficientnet_blocks.___torch_mangle_37.InvertedResidual,
    x: Tensor) -> Tensor:
    _0 = __torch__.timm.models.layers.drop.drop_path
    x0 = (self.conv_pw).forward(x, )
    x1 = (self.bn1).forward(x0, )
    x2 = (self.act1).forward(x1, )
    x3 = (self.conv_dw).forward(x2, )
    x4 = (self.bn2).forward(x3, )
    x5 = (self.act2).forward(x4, )
    x6 = (self.conv_pwl).forward(x5, )
    x7 = (self.bn3).forward(x6, )
    if self.has_residual:
      if torch.gt(self.drop_path_rate, 0.):
        x10 = _0(x7, self.drop_path_rate, self.training, )
        x9 = x10
      else:
        x9 = x7
      x8 = torch.add_(x9, x, alpha=1)
    else:
      x8 = x7
    return x8
