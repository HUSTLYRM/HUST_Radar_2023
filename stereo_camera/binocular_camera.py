import sys
from ctypes import *
import cv2
import numpy as np
from stereo_camera.MvImport import MvCameraControl_class as hk


# 清除摄像头缓存数据
def read_camera_cache(_cam):
    success = _cam.grab()
    while success:
        success = _cam.grab()


def camera_init(cfg):
    device_list = hk.MV_CC_DEVICE_INFO_LIST()

    # 枚举设备 只使用了usb相机，所以相应地传入第一个参数(区别于官方示例)
    # ret指返回值
    _ret = hk.MvCamera.MV_CC_EnumDevices(hk.MV_USB_DEVICE, device_list)
    if _ret != 0:
        print("enum devices fail! ret[0x%x]" % _ret)
        sys.exit()

    if device_list.nDeviceNum == 0:
        print("find no device!")
        sys.exit()

    print("Find %d devices!" % device_list.nDeviceNum)

    # 用于标记左右相机是否找到，并记录其编号
    n_connection_num = [-1, -1]

    for i in range(0, device_list.nDeviceNum):
        mvcc_dev_info = cast(device_list.pDeviceInfo[i], POINTER(hk.MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == hk.MV_USB_DEVICE:
            # 输出相机信息
            print("\nu3v device: [%d]" % i)
            str_mode_name = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                str_mode_name = str_mode_name + chr(per)
            print("device model name: %s" % str_mode_name)

            str_serial_number = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                str_serial_number = str_serial_number + chr(per)
            print("user serial number: %s" % str_serial_number)

            # 根据硬件id判断左右相机
            if str_serial_number.endswith(cfg['id']['left'][-2:]):
                n_connection_num[0] = i
            elif str_serial_number.endswith(cfg['id']['right'][-2:]):
                n_connection_num[1] = i

    if n_connection_num[0] == -1 or n_connection_num[1] == -1:
        print("less than 2 camera")
        sys.exit()

    # 创建相机实例
    _cam_left = hk.MvCamera()
    _cam_right = hk.MvCamera()

    # 分别为左右相机创建句柄
    st_device_list = cast(device_list.pDeviceInfo[int(n_connection_num[0])], POINTER(hk.MV_CC_DEVICE_INFO)).contents
    _ret = _cam_left.MV_CC_CreateHandle(st_device_list)
    if _ret != 0:
        print("create handle fail! ret[0x%x]" % _ret)
        sys.exit()

    st_device_list = cast(device_list.pDeviceInfo[int(n_connection_num[1])], POINTER(hk.MV_CC_DEVICE_INFO)).contents
    _ret = _cam_right.MV_CC_CreateHandle(st_device_list)
    if _ret != 0:
        print("create handle fail! ret[0x%x]" % _ret)
        sys.exit()

    # 打开设备 此时并不读取输入流
    _cam_left.MV_CC_OpenDevice(hk.MV_ACCESS_Exclusive, 0)
    _cam_right.MV_CC_OpenDevice(hk.MV_ACCESS_Exclusive, 0)

    # 清空原有缓存(可能存在)
    _cam_left.MV_CC_ClearImageBuffer()
    _cam_right.MV_CC_ClearImageBuffer()

    return _cam_left, _cam_right


def set_parameters(_cam=hk.MvCamera(), _name='', cfg=None):
    # 设置触发模式为on 用于时间同步
    if cfg is None:
        cfg = dict()
    _ret = _cam.MV_CC_SetEnumValue("TriggerMode", hk.MV_TRIGGER_MODE_ON)
    if _ret != 0:
        print("[%s] set trigger mode fail! ret[0x%x]" % (_name, _ret))
        sys.exit()
    # 设置触发源 设置触发源为软触发
    _ret = _cam.MV_CC_SetEnumValue("TriggerSource", hk.MV_TRIGGER_SOURCE_LINE2)
    # LINE2 hk.MV_TRIGGER_SOURCE_SOFTWARE
    if _ret != 0:
        print("[%s] set trigger source fail! ret[0x%x]" % (_name, _ret))
        sys.exit()

    # 设置尺寸
    _ret = _cam.MV_CC_SetIntValue("Width", cfg['param']['Width'])
    if _ret != 0:
        print("[%s] set width fail! ret[0x%x]" % (_name, _ret))
        sys.exit()
    _ret = _cam.MV_CC_SetIntValue("Height", cfg['param']['Height'])
    if _ret != 0:
        print("[%s] set height fail! ret[0x%x]" % (_name, _ret))
        sys.exit()

    # 设置垂直偏移 (1280*1024总区域相对于采样范围的上移，0时采样范围大概处于顶端部分)
    _ret = _cam.MV_CC_SetIntValue("OffsetY", cfg['param']['OffsetY'])
    if _ret != 0:
        print("[%s] set offset(y) fail! ret[0x%x]" % (_name, _ret))
        sys.exit()
    """if _name == 'left_camera':
        _ret = _cam.MV_CC_SetIntValue("OffsetX", 64)
        if _ret != 0:
            print("[%s] set offset(y) fail! ret[0x%x]" % (_name, _ret))
            sys.exit()"""

    # 设置曝光时间
    _ret = _cam.MV_CC_SetFloatValue("ExposureTime", cfg['param']['ExposureTime'])
    if _ret != 0:
        print("[%s] set exposure fail! ret[0x%x]" % (_name, _ret))
        sys.exit()

    # 设置gamma矫正
    _ret = _cam.MV_CC_SetBoolValue("GammaEnable", cfg['param']['GammaEnable'])
    if _ret != 0:
        print("[%s] set gammaEnable fail! ret[0x%x]" % (_name, _ret))
        sys.exit()
    _ret = _cam.MV_CC_SetFloatValue("Gamma", cfg['param']['Gamma'])
    if _ret != 0:
        print("[%s] set gamma fail! ret[0x%x]" % (_name, _ret))
        sys.exit()

    # 设置 白平衡，用于图像颜色“不正”时
    _ret = _cam.MV_CC_SetEnumValueByString("BalanceWhiteAuto", cfg['param']['BalanceWhiteAuto'])
    if _ret != 0:
        print("[%s] set balanceWhiteAuto fail! ret[0x%x]" % (_name, _ret))
        sys.exit()
    _ret = _cam.MV_CC_SetEnumValueByString("BalanceRatioSelector", "Red")
    if _ret != 0:
        print("[%s] set balanceWhiteRatio:Red fail! ret[0x%x]" % (_name, _ret))
        sys.exit()
    _ret = _cam.MV_CC_SetIntValue("BalanceRatio", cfg['param']['BalanceRatioR'])
    if _ret != 0:
        print("[%s] set balanceRatio fail! ret[0x%x]" % (_name, _ret))
        sys.exit()
    _ret = _cam.MV_CC_SetEnumValueByString("BalanceRatioSelector", "Green")
    if _ret != 0:
        print("[%s] set balanceWhiteRatio:Green fail! ret[0x%x]" % (_name, _ret))
        sys.exit()
    _ret = _cam.MV_CC_SetIntValue("BalanceRatio", cfg['param']['BalanceRatioG'])
    if _ret != 0:
        print("[%s] set balanceRatio fail! ret[0x%x]" % (_name, _ret))
        sys.exit()
    _ret = _cam.MV_CC_SetEnumValueByString("BalanceRatioSelector", "Blue")
    if _ret != 0:
        print("[%s] set balanceWhiteRatio:Blue fail! ret[0x%x]" % (_name, _ret))
        sys.exit()
    _ret = _cam.MV_CC_SetIntValue("BalanceRatio", cfg['param']['BalanceRatioB'])
    if _ret != 0:
        print("[%s] set balanceRatio fail! ret[0x%x]" % (_name, _ret))
        sys.exit()


def camera_close(_cam, name='camera'):
    _cam.MV_CC_StopGrabbing()

    # 关闭设备
    _ret = _cam.MV_CC_CloseDevice()
    if _ret != 0:
        print("close device fail! ret[0x%x]" % _ret)
        sys.exit()
    if _ret == 0:
        print(name + ' closed')
    # 销毁句柄
    _ret = _cam.MV_CC_DestroyHandle()
    if _ret != 0:
        print("destroy handle fail! ret[0x%x]" % _ret)
        sys.exit()


def to_seconds(_h, _l):
    return (int(_h) & 0xffffffff) + (float(_l & 0xffffffff)) / pow(2, 32)


def get_frame(_cam=hk.MvCamera(), _name='', ret=POINTER(c_bool)):
    frame = hk.MV_FRAME_OUT()
    memset(byref(frame), 0, sizeof(frame))
    ret.contents = False

    # 读取图像
    _ret = _cam.MV_CC_GetImageBuffer(frame, 100)
    frame_info = frame.stFrameInfo

    if _ret == hk.MV_OK:
        ret.contents = True

        print("[%s] get one frame: Width[%d], Height[%d], nFrameNum[%d], timestamp(high)[%d], timestamp(low)[%d]"
              % (_name, int(frame_info.nWidth), int(frame_info.nHeight),
                 int(frame_info.nFrameNum), int(frame_info.nDevTimeStampHigh), int(frame_info.nDevTimeStampLow)))

        b1 = hk.MVCC_FLOATVALUE()

        _cam.MV_CC_GetFloatValue('Brightness', b1)
        print(b1.fCurValue)

        # 像素格式转换
        channel_num = 3
        buffer = (c_ubyte * (frame_info.nWidth * frame_info.nHeight * channel_num))()
        size = frame_info.nWidth * frame_info.nHeight * channel_num
        st_convert_param = hk.MV_CC_PIXEL_CONVERT_PARAM()
        st_convert_param.nWidth = frame_info.nWidth  # 图像宽
        st_convert_param.nHeight = frame_info.nHeight  # 图像高
        st_convert_param.pSrcData = frame.pBufAddr  # 输入数据缓存
        st_convert_param.nSrcDataLen = frame_info.nFrameLen  # 输入数据大小
        st_convert_param.enSrcPixelType = frame_info.enPixelType  # 输入像素格式
        st_convert_param.enDstPixelType = hk.PixelType_Gvsp_BGR8_Packed  # 输出像素格式
        st_convert_param.pDstBuffer = buffer  # 输出数据缓存
        st_convert_param.nDstBufferSize = size  # 输出缓存大小
        _cam.MV_CC_ConvertPixelType(st_convert_param)

        # 转为OpenCV可以处理的numpy数组 (Mat)
        image = np.asarray(buffer).reshape((frame_info.nHeight, frame_info.nWidth, 3))
        _cam.MV_CC_FreeImageBuffer(frame)

        return image

    else:
        print("[%s] no data[0x%x]" % (_name, _ret))
        return None


def get_camera(cfg=None):
    # cameras init
    # 寻找、连接并开启相机 (获取句柄)
    if cfg is None:
        cfg = dict()
    cam_left, cam_right = camera_init(cfg)

    print(cam_left.MV_CC_IsDeviceConnected())
    print(cam_right.MV_CC_IsDeviceConnected())

    # 设置参数
    set_parameters(cam_left, 'left camera', cfg)
    set_parameters(cam_right, 'right camera', cfg)

    # 开始取流(可以理解为打开通道，此时并无图像传输)
    ret = cam_left.MV_CC_StartGrabbing()
    if ret != 0:
        print("left 开始取流失败! ret[0x%x]" % ret)
        sys.exit()

    ret = cam_right.MV_CC_StartGrabbing()
    if ret != 0:
        print("right 开始取流失败! ret[0x%x]" % ret)
        sys.exit()

    ret_p = POINTER(c_bool)
    ret_p.contents = False
    ret_q = POINTER(c_bool)
    ret_q.contents = False

    return cam_left, cam_right, ret_p, ret_q


def get_video_loader(video_path):
    """size_dont_fit = False"""
    cam = cv2.VideoCapture(video_path)

    frames_total = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    print("frames_total: ", frames_total)

    fps = cam.get(cv2.CAP_PROP_FPS)
    size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    return cam, fps, size
