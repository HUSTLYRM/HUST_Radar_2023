class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.timm.models.efficientnet_blocks.___torch_mangle_48.InvertedResidual
  __annotations__["1"] = __torch__.timm.models.efficientnet_blocks.___torch_mangle_53.InvertedResidual
  __annotations__["2"] = __torch__.timm.models.efficientnet_blocks.___torch_mangle_53.InvertedResidual
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_54.Sequential,
    input: Tensor) -> Tensor:
    _0 = getattr(self, "0")
    _1 = getattr(self, "1")
    _2 = getattr(self, "2")
    input0 = (_0).forward(input, )
    input1 = (_1).forward(input0, )
    return (_2).forward(input1, )
  def __len__(self: __torch__.torch.nn.modules.container.___torch_mangle_54.Sequential) -> int:
    return 3
