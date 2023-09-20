# 双目相机模块

![Screenshot 2023-07-15 154616](https://github.com/HUSTLYRM/HUST_Radar_2023/blob/main/images/bincam.png)

## 简述

使用海康工业相机+单片机硬触发+深度学习立体匹配完成，核心是开源立体匹配算法[Correlate-and-Excite](https://github.com/antabangun/coex)，该算法是当时找到的唯一一个满足实时性要求的，但在两张1280*720的图像上帧率仍不满10fps（但考虑到发送频率限制，已经足够了）。



## 其他说明

* 海康似乎没有专门的python文档，“MvImport”目录下是从示例程序中整理的依赖文件。
* 由于CoEx模型的设计（“关联性”），有时“初始化”似乎也会影响效果。
* 对标定的要求较高（对外参精度要求高），我使用的是matlab的标定app，一般使用200-300对有效图片（覆盖各种角度和距离）时效果比较好。不知道是否合理。

