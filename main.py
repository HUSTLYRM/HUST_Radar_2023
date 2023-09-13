print('initializing...')
import cv2
import os
import shutil
import sys
import csv
import time
import numpy as np
import torch
from ruamel.yaml import YAML
from ultralytics import YOLO

import my_serial as messager
from stereo_camera import binocular_camera as bc
from stereo_camera.coex_matcher import CoExMatcher
from target import Targets
from anchor import Anchor
from anchor import set_by_hand
from macro import *
import coord_solver as cc
from utils.chessboard_corner import find_chessboard_corners

from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import gui
print('[main] modules imported')

CarsTotal = 5
RedCarsID = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
BlueCarsID = {1: 101, 2: 102, 3: 103, 4: 104, 5: 105}
classes = ["car", "armor1red", "armor2red", "armor3red", "armor4red", "armor5red",
           "armor1blue", "armor2blue", "armor3blue", "armor4blue", "armor5blue", "base", "ignore"]

ENEMY_COLOR = BLUE
portx = 'COM3'
device = torch.device('cuda:0')

main_cfg_path = "./configs/main_config.yaml"
binocular_camera_cfg_path = "./configs/bin_cam_config.yaml"
monocular_camera_cfg_path = "./configs/mon_cam_config.yaml"
main_cfg = YAML().load(open(main_cfg_path, encoding='Utf-8', mode='r'))
bin_cam_cfg = YAML().load(open(binocular_camera_cfg_path, encoding='Utf-8', mode='r'))
mon_cam_cfg = YAML().load(open(monocular_camera_cfg_path, encoding='Utf-8', mode='r'))

if main_cfg['debug']:
    portx = 'COM7'

arc_roll = bin_cam_cfg['set']['roll'] * np.pi / 180
cos_arc_roll = np.cos(arc_roll)
cam_bias = bin_cam_cfg['set']['bias']

if ENEMY_COLOR == RED:  # self = BLUE
    Enemy_Car_List = [1, 2, 3, 4, 5]
    Own_Car_List = [101, 102, 103, 104, 105]
    Encirclement_List = [101, 103, 104, 105, 107]
    Guard = 107
    Radar = 109
    label, seed, buf = 1, 11200, 14150
    radar_base = [28988.19, 6017.49, 2500]  # blue base
    cam_pos = [radar_base[0] + cam_bias[0], radar_base[1] - cam_bias[1], radar_base[2] + cam_bias[2]]
    sign = -1
    ALLY_COLOR = BLUE

elif ENEMY_COLOR == BLUE:  # self = RED
    Enemy_Car_List = [101, 102, 103, 104, 105]
    Own_Car_List = [1, 2, 3, 4, 5]
    Encirclement_List = [1, 3, 4, 5, 7]
    Guard = 7
    Radar = 9
    label, seed, buf = 101, 17942, 1799
    radar_base = [-987.55, 9018.02, 2500]  # red base
    cam_pos = [radar_base[0] - cam_bias[0], radar_base[1] + cam_bias[1], radar_base[2] + cam_bias[2]]
    sign = 1
    ALLY_COLOR = RED


targets = Targets(ENEMY_COLOR)
Enemy_Cars = []
count_down = 420
blood = [0 for i in range(16)]
guard_location = (0, 0)


exit_signal = False
Loop = True

ser = messager.serial_init(portx)

# 数据保存
if main_cfg['ctrl']['SAVE_IMG']:
    time_now = time.localtime()
    img_folder = './record' + str(time_now[1]) + '-' + str(time_now[2]) + '-' + str(time_now[3]) + '-' + str(
        time_now[4])  # months + days + hours
    primal_folder = img_folder + '/primal'
    result_folder = img_folder + '/result'
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
        os.mkdir(primal_folder)
        os.mkdir(result_folder)

if main_cfg['ctrl']['RECORDING']:
    time_now = time.localtime()
    video_folder = './video_record' + str(time_now[1]) + '-' + str(time_now[2]) + '-' + \
                   str(time_now[3]) + '-' + str(time_now[4]) + '-' + str(time_now[5]) + \
                   '-' + str(time_now[6])  # months-days-hours-mins-secs
    raw_video_folder = video_folder + '/raw'
    left_video_folder = raw_video_folder + '/left'
    right_video_folder = raw_video_folder + '/right'
    lf_video_folder = raw_video_folder + '/lf'
    # result_video_folder = video_folder + '/result'
    if not os.path.exists(video_folder):
        os.mkdir(video_folder)
    if not os.path.exists(raw_video_folder):
        os.mkdir(raw_video_folder)
    if not os.path.exists(left_video_folder):
        os.mkdir(left_video_folder)
    if not os.path.exists(right_video_folder):
        os.mkdir(right_video_folder)
    if not os.path.exists(lf_video_folder):
        os.mkdir(lf_video_folder)
    if not os.path.exists(video_folder + '/cfg'):
        os.mkdir(video_folder + '/cfg')
    dst = video_folder + '/cfg'
    shutil.copy(binocular_camera_cfg_path, dst)
    shutil.copy(monocular_camera_cfg_path, dst)

    fourcc_colored = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ('M', 'P', '4', 'V')
    frame_size = (bin_cam_cfg['param']['Width'], bin_cam_cfg['param']['Height'])
    left_video = cv2.VideoWriter(left_video_folder + "/raw_left.mp4", fourcc_colored, 12, frame_size, True)
    right_video = cv2.VideoWriter(right_video_folder + "/raw_right.mp4", fourcc_colored, 12, frame_size, True)
    depth_video = cv2.VideoWriter(video_folder + "/dep_view_left.mp4", fourcc_colored, 12, frame_size, True)
    lf_video = cv2.VideoWriter(lf_video_folder + "/lf.mp4", fourcc_colored, 12, frame_size, True)
    fourcc_grey = cv2.VideoWriter_fourcc('d', 'i', 'v', 'x')
    disp_video = cv2.VideoWriter(video_folder + "/disp_left.divx", fourcc_grey, 12, frame_size, True)

if main_cfg['ctrl']['SAVE_CSV']:
    header = ['car_center_x', 'car_center_y', 'x', 'y', 'z']
    chart = open("data.csv", "w", newline='')
    writer = csv.DictWriter(chart, header)
    writer.writeheader()


# with qt
def push_button_clicked_quit():
    global exit_signal, Loop
    exit_signal = True
    print('exit')
    Loop = False


def main():
    global targets
    camera_left = camera_right = camera_lf = None
    ret_p = ret_q = None
    coex_matcher = None
    model_car = None
    monitor = None

    main_scene = None
    timeout_draw_zone = False

    if main_cfg['ctrl']['MODE'] == 'video':
        camera_left, fps, size = bc.get_video_loader(main_cfg['video'])

    elif main_cfg['ctrl']['MODE'] == 'camera':
        print("\nLoading binocular camera")
        camera_left, camera_right, ret_p, ret_q = bc.get_camera(bin_cam_cfg)
        print("Done")

        print("\nLoading matching model")
        coex_matcher = CoExMatcher(bin_cam_cfg)
        print("Done\n")

    left_cam_cfg = dict()
    left_cam_cfg['intrinsic'] = bin_cam_cfg['calib']['intrinsic1']
    left_cam_cfg['distortion'] = bin_cam_cfg['calib']['distortion1']
    camera_pose_solver = cc.CameraPoseSolver(ALLY_COLOR, left_cam_cfg)
    if main_cfg['ctrl']['ANCHOR']:
        anchor = Anchor()
        while True:  # this while is for case where no img got
            image_left = bc.get_frame(camera_left, 'left_camera', ret_p)
            # to keep synchronous
            image_right = bc.get_frame(camera_right, 'right_camera', ret_q)

            if image_left is None or image_right is None:
                continue
            set_by_hand(image_left, anchor)
            camera_pose_solver.init_by_anchor(anchor)
            break
    else:
        # camera_pose_solver.init_by_constant()
        pass

    if main_cfg['ctrl']['DETECT']:
        print('Loading Car Model')
        model_car = YOLO(main_cfg['weights']['yolov8'])
        print('Done\n')

    if main_cfg['ctrl']['GUI']:
        print('preparing gui')
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        app = QApplication(sys.argv)

        MainWindow = QMainWindow()
        monitor = gui.Ui_Monitor()
        monitor.setupUi(MainWindow)
        monitor.pushButton_quit.clicked.connect(push_button_clicked_quit)
        main_scene = QGraphicsScene()
        monitor.main_frame_view.setScene(main_scene)
        monitor.main_frame_view.show()
        MainWindow.show()
        print('done')

        if timeout_draw_zone:
            monitor.debug_msg.append('zone initialize: timeout, using saved one\n')

    cnt = 0
    start = 0.
    last = time.time()
    # Here the main loop
    global Loop
    while Loop:
        if cv2.waitKey(1) == ord('q'):
            Loop = False
        image_left = bc.get_frame(camera_left, 'left_camera', ret_p)
        image_right = bc.get_frame(camera_right, 'right_camera', ret_q)

        if image_right is not None and image_left is not None:
            if main_cfg['ctrl']['RECORDING']:
                left_video.write(image_left)
                right_video.write(image_right)

            re_left, point_cloud, disp_np = coex_matcher.inference(image_left, image_right)

            if main_cfg['ctrl']['RECORDING']:
                disp_video.write(disp_np)
                cv2.imshow('raw_disp', disp_np)

            disp_np = cv2.applyColorMap(2 * disp_np, cv2.COLORMAP_MAGMA)
            if main_cfg['ctrl']['RECORDING']:
                depth_video.write(disp_np)

            if main_cfg['ctrl']['CHESSBOARD']:
                pattern_size = (8, 6)
                corners, image_with_corners = find_chessboard_corners(re_left, pattern_size)
                # print(corners)
                if corners is not None:
                    cv2.putText(image_with_corners,
                                '[ ' + str(
                                    round(point_cloud[int(corners[0][0][1])][int(corners[0][0][0])][0], 2)) + ', '
                                + str(round(point_cloud[int(corners[0][0][1])][int(corners[0][0][0])][1], 2)) + ', '
                                + str(round(point_cloud[int(corners[0][0][1])][int(corners[0][0][0])][2], 2)) + ']',
                                (int(corners[0][0][0]), int(corners[0][0][1])),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.7,
                                color=(255, 255, 255),
                                thickness=1,
                                lineType=cv2.LINE_AA
                                )
                    cv2.imshow('xyz', image_with_corners)

            if cnt == 10:
                start = time.time()
            if cnt > 10:
                now = time.time()
                fps = (cnt - 10) / (now - start)
                cv2.putText(disp_np,
                            "fps: " + "%.2f" % fps,
                            (4, 40),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.9,
                            color=(255, 255, 255),
                            thickness=2,
                            lineType=cv2.LINE_AA
                            )
            cv2.imshow('disp', disp_np)

            if main_cfg['ctrl']['DETECT']:
                dst_img = np.copy(re_left)

                result = model_car.predict(dst_img, show=True)
                boxes = result[0].boxes.data.cpu()
                boxes = boxes.numpy()

                print(boxes)
                targets.update(boxes)
                for target in targets.targets:
                    if target.conf > 0:
                        cam_coord = [[point_cloud[int(target.center_yx[0])][int(target.center_yx[1])][0]],
                                     [point_cloud[int(target.center_yx[0])][int(target.center_yx[1])][1]],
                                     [point_cloud[int(target.center_yx[0])][int(target.center_yx[1])][2]]]
                        field_coord = camera_pose_solver.get_field_coord(cam_coord)
                        target.x = field_coord[0]
                        target.y = field_coord[1]
                        if main_cfg['debug']:
                            msg = str(cam_coord)
                            cv2.putText(re_left,
                                        msg,
                                        (int(target.center_yx[1]), int(target.center_yx[0])),
                                        cv2.FONT_HERSHEY_PLAIN,
                                        1.0,
                                        (0, 0, 255),
                                        thickness=1)
                if main_cfg['debug']:
                    cv2.imshow('dist', re_left)

                for car in targets.targets:
                    if car.conf > 0:
                        now = time.time()
                        # 距离上一次发送时间小于0.1s:sleep
                        if now - last < 0.1:
                            time.sleep(0.1 - (now - last))
                        messager.send_enemy_location(ser, car.get_id(), car.x / 1000,
                                                     car.y / 1000)  # mm to m
                        if main_cfg['ctrl']['GUI']:
                            monitor.debug_msg.append('at [' + str(now) + '] send: ' + str(car.get_id()) + ' ' +
                                                     str(car.x / 1000) + ' ' + str(car.y / 1000) + '\n')
                        last = time.time()
            else:
                cv2.imshow('re_left', re_left)

            if main_cfg['ctrl']['GUI']:
                resized = cv2.resize(re_left, (720, 360))
                frame = QImage(resized, 720, 360, 720 * 3, QImage.Format_BGR888)
                main_scene.clear()  # 先清空上次的残留
                pixel_map = QPixmap.fromImage(frame)
                main_scene.addPixmap(pixel_map)

        cnt += 1
    '- end of loop -----------------------------------------------------------------------------'

    if main_cfg['ctrl']['MODE'] == 'camera':
        bc.camera_close(camera_left, 'camera_left')
        bc.camera_close(camera_right, 'camera_right')
        bc.camera_close(camera_lf, 'camera_lf')

    cv2.destroyAllWindows()

    if main_cfg['ctrl']['RECORDING']:
        os.system('copy bin_cam_config.yaml ' + video_folder[2:] + '/cfg/bin_cam_config.yaml')
        os.system('copy mon_cam_config.yaml ' + video_folder[2:] + '/cfg/mon_cam_config.yaml')
        os.system('copy main_config.yaml ' + video_folder[2:] + '/cfg/main_config.yaml')
        left_video.release()
        right_video.release()
        disp_video.release()
        depth_video.release()
        lf_video.release()

    if main_cfg['ctrl']['SAVE_CSV']:
        chart.close()
    print('release!')


if __name__ == '__main__':
    main()
