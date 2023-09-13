class Stereo(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  _dtype : int
  _device : Device
  exp_save_path : None
  trainer : None
  _distrib_type : None
  _device_type : None
  use_amp : bool
  precision : int
  _example_input_array : None
  _datamodule : None
  _results : None
  _current_fx_name : str
  _running_manual_backward : bool
  _current_hook_fx_name : None
  _current_dataloader_idx : None
  running_stage : None
  _automatic_optimization : bool
  dataname : None
  stereo : __torch__.models.stereo.CoEx.CoEx
  def __automatic_optimization_getter(self: __torch__.stereo.Stereo) -> bool:
    return self._automatic_optimization
  def __automatic_optimization_setter(self: __torch__.stereo.Stereo,
    automatic_optimization: bool) -> None:
    self._automatic_optimization = automatic_optimization
    return None
  def forward(self: __torch__.stereo.Stereo,
    im: Tensor) -> Tensor:
    _0 = torch.slice(im, 0, 0, 9223372036854775807, 1)
    imgL = torch.slice(_0, 1, 0, 3, 1)
    _1 = torch.slice(im, 0, 0, 9223372036854775807, 1)
    imgR = torch.slice(_1, 1, 3, 9223372036854775807, 1)
    _2 = torch.slice(torch.size(imgL), -2, 9223372036854775807, 1)
    h, w, = _2
    h_pad = torch.sub(32, torch.remainder(h, 32))
    w_pad = torch.sub(32, torch.remainder(w, 32))
    imgL0 = __torch__.torch.nn.functional._pad(imgL, [0, w_pad, 0, h_pad], "constant", 0., )
    imgR0 = __torch__.torch.nn.functional._pad(imgR, [0, w_pad, 0, h_pad], "constant", 0., )
    _3 = (self.stereo).forward(imgL0, imgR0, )
    _4 = torch.slice(_3[0], 0, 0, 9223372036854775807, 1)
    disp_pred = torch.slice(torch.slice(_4, 1, 0, h, 1), 2, 0, w, 1)
    return disp_pred
