class channelAtt(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  im_att : __torch__.torch.nn.modules.container.___torch_mangle_84.Sequential
  def forward(self: __torch__.models.stereo.submodules.utils.___torch_mangle_85.channelAtt,
    cv: Tensor,
    im: Tensor) -> Tensor:
    channel_att = torch.unsqueeze((self.im_att).forward(im, ), 2)
    cv0 = torch.mul(torch.sigmoid(channel_att), cv)
    return cv0
