# HUST_Radar_2023

    华中科技大学狼牙战队在23赛季使用的雷达站代码开源。

## 简述

​	  不同于使用较为广泛的单相机投影/PnP方案和多相机+激光雷达联合标定方案，本赛季我们采用“自制”双目相机完成定位功能、使用两个工业相机搭建双目相机进行测距。

​	  对于双目相机，尝试了一系列立体匹配算法，对双目标定细节以及测距效果的定量测试方案也进行了探索。使用YOLOv8进行整车和装甲板的检测，并尝试优化小目标检测效果、增加跟踪器。对于程序整体效果，设计了可视化接收进程，方便程序的调试。

​	  很多地方都不完善，更多的是希望能提供一些参考。



## 文件结构

```
HUST_Radar_2023
├─camera_locater 相机位姿解算
├─configs 配置文件
├─enemy_locater 目标定位
├─gui 调试界面
├─image README用
├─stereo_camera 双目相机
│  ├─ckpts
│  ├─coex
│  ├─configs
│  ├─MvImport
│  ├─MvSdkLog
└─utils 测试工具
```

## 运行效果

​	  双目测距上限理论上很高，但最终实现的效果不尽人意。

### 整体效果

​	  主界面如下。其中公路区计划用单独的长焦相机监视，但后来代码实现有问题，该部分未在赛场上使用。小地图示意、动态参数调整、调试信息输出未完成。

<img src=".\images\main.png" alt="main" style="zoom: 34%;" />

​	  部分其他调试输出如下。

<img src=".\images\info.png" alt="info" style="zoom: 27%;" />

### 测距效果

​	  测试方案：在已知位置放置二维码，由双目相机得到点云后，检测二维码并读取其坐标，与真值比较。

<img src=".\images\plot.png" alt="plot" style="zoom: 67%;" />

​	  测试仍不充分。测试结果并不精确，二维码在双目可能的有效测距范围内就已经难以识别，且早期测试在狭窄走廊进行，由于选择的立体匹配模型特性，其结果受“地形“影响很大（如墙壁），实际在赛场看到的效果似乎优于测试效果。

<img src=".\images\disp_cmp.png" alt="disp_cmp" style="zoom:30%;" />

### 目标检测效果



### 地图数据可视化

    调试模块：[RadarDebugger_Monitor](https://github.com/HUSTLYRM/RadarDebugger_Monitor)

<img src=".\images\debug.png" alt="debug" style="zoom: 30%;" />



## 运行流程

    各模块内部细节见模块内README文件说明

todo：流程图





## 硬件平台

#### 算力端

CPU：12th Gen Intel(R) Core(TM) i7-12800HX

显卡：NVIDIA GeForce RTX 3080 Ti Laptop GPU

#### 传感器端

相机：海康MV-CA013-20UC * 2 （另有一个同型号相机但未使用）

此外，用单片机通过触发线实现对左右相机的同步触发。

## 安装

### 依赖

​	  程序基于Python3.9完成，依赖较多（建议使用虚拟环境），请参考requirements.txt。其中Correlate-and-Excite（实时立体匹配，CoEx）和YOLOv8的依赖请参考官方文档。

* **注意先满足立体匹配模型的依赖，再添加其他包**，此外若同样使用海康工业相机，需要先安装海康SDK。
* 用到的“serial”模块是“pyserial”。

## 功能/操作说明 

### 参数配置

main_config：主要用于开关程序功能，如录制、识别等是否启用。此外还包含了权重文件、视频路径的配置。

bin_camera_config：用于输入双目相机内外参，以及对工业相机的图像采集进行配置。

### 运行

Windows运行脚本可参考[launch.bat](./launch.bat)

### 调试

​	  程序具有视频调试模式，通过在main_config的ctrl条目配置MODE为“video”、配置左右视图视频路径来脱离相机进行调试。

​	  可以通过配置main_config的ctrl条目中的其他选项，选择性地开关相应模块功能，对模块单独调试。

​	  此外可以使用虚拟串口工具在本地生成串口对，同时运行本程序和上文提到的地图可视化程序，来帮助调试。

## 相关资源

[YOLOv8权重]()

[双视图比赛视频]()

## 局限性

​	  从上文效果演示也可以看出，本程序很多地方都不完善。

### 双目相机

#### 像素与性能

#### 机械精度与标定

### 小目标检测

### 其他

​	  此外，信息收发逻辑、各种前后处理的等各种细节都有很大的改进空间；一些简单的决策辅助功能也没有实现，如告知哨兵目标的大致方位、是否需要抬头等。

## 特别致谢

​	初次参与到比赛中，雷达站又只有一个人，我在技术和沟通上都遇到了很大的困难。非常感谢队长、副队和雷达站的前辈在各方面的支持，感谢曾老师的指导和建议（虽然我最终没能做好），以及队友们的帮助（大量）。

​	感谢Antyanta Bangunharcana和他的开源项目[Correlate-and-Excite](https://github.com/antabangun/coex)，让我的想法有机会登上赛场。

​	感谢所有的开源贡献者，我从其他开源项目中学到了很多、也借用了很多。
