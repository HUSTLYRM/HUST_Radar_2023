class Feature(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  type : str
  conv_stem : __torch__.torch.nn.modules.conv.Conv2d
  bn1 : __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  act1 : __torch__.torch.nn.modules.activation.ReLU6
  block0 : __torch__.torch.nn.modules.container.___torch_mangle_3.Sequential
  block1 : __torch__.torch.nn.modules.container.___torch_mangle_15.Sequential
  block2 : __torch__.torch.nn.modules.container.___torch_mangle_25.Sequential
  block3 : __torch__.torch.nn.modules.container.___torch_mangle_44.Sequential
  block4 : __torch__.torch.nn.modules.container.___torch_mangle_55.Sequential
  def forward(self: __torch__.models.stereo.submodules.feature.Feature,
    x: Tensor) -> Tuple[Tensor, List[Tensor]]:
    _0 = self.act1
    _1 = (self.bn1).forward((self.conv_stem).forward(x, ), )
    x0 = (_0).forward(_1, )
    x2 = (self.block0).forward(x0, )
    x4 = (self.block1).forward(x2, )
    x8 = (self.block2).forward(x4, )
    x16 = (self.block3).forward(x8, )
    x32 = (self.block4).forward(x16, )
    x_out = [x4, x8, x16, x32]
    return (x2, x_out)
class FeatUp(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  type : str
  deconv32_16 : __torch__.models.stereo.submodules.util_conv.Conv2x
  deconv16_8 : __torch__.models.stereo.submodules.util_conv.___torch_mangle_62.Conv2x
  deconv8_4 : __torch__.models.stereo.submodules.util_conv.___torch_mangle_68.Conv2x
  conv4 : __torch__.models.stereo.submodules.util_conv.___torch_mangle_67.BasicConv
  def forward(self: __torch__.models.stereo.submodules.feature.FeatUp,
    x4: Tensor,
    x8: Tensor,
    x16: Tensor,
    x32: Tensor,
    y4: Tensor,
    y8: Tensor,
    y16: Tensor,
    y32: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
    x160 = (self.deconv32_16).forward(x32, x16, )
    y160 = (self.deconv32_16).forward(y32, y16, )
    x80 = (self.deconv16_8).forward(x160, x8, )
    y80 = (self.deconv16_8).forward(y160, y8, )
    x40 = (self.deconv8_4).forward(x80, x4, )
    y40 = (self.deconv8_4).forward(y80, y4, )
    x41 = (self.conv4).forward(x40, )
    y41 = (self.conv4).forward(y40, )
    _2 = ([x41, x80, x160, x32], [y41, y80, y160, y32])
    return _2
