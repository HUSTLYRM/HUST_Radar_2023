# 相机位姿估计模块 camera_locator

​		初版直接把雷达站放在预定位置，后来为了提高精度而使用场地标签。原本考虑视觉识别R0、B0标签以完成相机定位，后来为了减小不确定性，采用手动取点的方式。

## 场地标签取点器 point_picker 

  ### 操作说明

| 功能     | 操作                           |
| -------- | ------------------------------ |
| 取点     | 鼠标左键点击                   |
| 缩放画面 | 鼠标滚轮（前滚放大、后滚缩小） |
| 拖拽画面 | （放大后），鼠标右键按下拖拽   |
| 撤回一点 | 按键 'd'                       |
| 结束取点 | 按键 'q'                       |

默认只取四点，从左上角逆时针选取。实际可以自行修改定义。

若不小心退出，当点不足4个时会自动恢复取点界面（可能有bug）

### 示例

<img src="..\images\picker.png" alt="picker" width="700"/>
