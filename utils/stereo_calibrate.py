import cv2
import os
import time
from ruamel.yaml import YAML
from stereo_camera import binocular_camera as bc

binocular_camera_cfg_path = "bin_cam_config.yaml"
bin_cam_cfg = YAML().load(open(binocular_camera_cfg_path, encoding='Utf-8', mode='r'))

title = format("calib_w%d_h%d_blx%d_bly%d_brx%d_bry%d_@%d" % (1280, 640, 0, 256, 0, 256, time.time()))
ir = "./stereo_camera/calib/{}".format(title)
if not os.path.exists(ir):
    os.mkdir(ir)
if not os.path.exists(ir + "/left"):
    os.mkdir(ir + "/left")
if not os.path.exists(ir + "/right"):
    os.mkdir(ir + "/right")
suffix = ".png"

print("\nLoading camera")
camera_left, camera_right, ret_p, ret_q = bc.get_camera(bin_cam_cfg)
print("Done")

cnt = 0
Loop = True
while Loop:
    image_left = bc.get_frame(camera_left, 'left_camera', ret_p)
    image_right = bc.get_frame(camera_right, 'right_camera', ret_q)

    cv2.imshow('left', image_left)
    cv2.imshow('right', image_right)

    cnt += 1
    cv2.waitKey(1500)  # 1500
    if ret_p.contents:
        cv2.imwrite(ir + "/left/" + str(cnt) + suffix, image_left)
    if ret_q.contents:
        cv2.imwrite(ir + "/right/" + str(cnt) + suffix, image_right)