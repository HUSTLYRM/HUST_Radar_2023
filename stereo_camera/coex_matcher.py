import cv2
import numpy as np
import torch
import time
from ruamel.yaml import YAML

from stereo_camera.coex.stereo import Stereo
# from stereo_camera.coex.dataloaders import KITTIRawLoader as KRL
from stereo_camera.coex.dataloaders.stereo import preprocess


torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = 'coex.yaml'
version = 0  # CoEx
half_precision = True


class CoExMatcher:
    def __init__(self, cam_cfg):
        self.cfg = None
        self.load_configs('./stereo_camera/configs/stereo/{}'.format(config))
        self.pose_ssstereo = Stereo.load_from_checkpoint(self.cfg['stereo_ckpt'],
                                                         strict=False,
                                                         cfg=self.cfg).cuda()
        self.pose_ssstereo.eval()
        """self.model = torch.jit.load("./coex/zoo/torchscript/CoEx.pt", map_location="cuda:0")
        self.model.eval()"""
        self.processed = preprocess.get_transform(augment=False)

        # remap part init
        # camera1
        self.intrinsic1 = np.array(cam_cfg['calib']['intrinsic1'])
        self.distortion1 = np.array(cam_cfg['calib']['distortion1'])
        # camera2
        self.intrinsic2 = np.array(cam_cfg['calib']['intrinsic2'])
        self.distortion2 = np.array(cam_cfg['calib']['distortion2'])
        self.image_size = (cam_cfg['param']['Width'], cam_cfg['param']['Height'])

        # shared
        self.R = np.array(cam_cfg['calib']['R'])
        self.T = np.array(cam_cfg['calib']['T'])
        self.R1, self.R2, self.P1, self.P2, self.Q, v1, v2 = cv2.stereoRectify(
            cameraMatrix1=self.intrinsic1, distCoeffs1=self.distortion1,
            cameraMatrix2=self.intrinsic2, distCoeffs2=self.distortion2,
            imageSize=self.image_size, R=self.R, T=self.T, flags=0, alpha=0)

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.intrinsic1, self.distortion1, self.R1, self.P1,
            self.image_size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.intrinsic2, self.distortion2, self.R2, self.P2,
            self.image_size, cv2.CV_32FC1)

    def inference(self, left_img, right_img):
        left = cv2.remap(left_img, self.map1x, self.map1y, cv2.INTER_LINEAR)
        right = cv2.remap(right_img, self.map2x, self.map2y, cv2.INTER_LINEAR)

        left_img_p = self.processed(left)
        right_img_p = self.processed(right)

        left_img_p = torch.unsqueeze(left_img_p, 0)
        right_img_p = torch.unsqueeze(right_img_p, 0)

        imgL, imgR = left_img_p.cuda(), right_img_p.cuda()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=half_precision):

                img = torch.cat([imgL, imgR], 0)
                disp = self.pose_ssstereo(img)

        disp_np = (disp[0]).data.cpu().numpy().astype(np.uint8)  # 2 *
        point_cloud = None
        point_cloud = cv2.reprojectImageTo3D(disp_np, self.Q, handleMissingValues=True, ddepth=cv2.CV_16S)

        return left, point_cloud, disp_np

    def load_configs(self, path):
        self.cfg = YAML().load(open(path, 'r'))
        backbone_cfg = YAML().load(
            open(self.cfg['model']['stereo']['backbone']['cfg_path'], 'r'))
        self.cfg['model']['stereo']['backbone'].update(backbone_cfg)

        ckpt = './stereo_camera/ckpts/coex/checkpoints/last.ckpt'
        self.cfg['stereo_ckpt'] = ckpt

        self.cfg['training']['th'] = 0
        self.cfg['training']['tw'] = 0


IMG = True
VIDEO = False
RECORDING = False
if __name__ == '__main__':
    if IMG:
        path_to_left_rgb_image_file = './img/distance/left/l.png'  # im2
        path_to_right_rgb_image_file = './img/distance/right/r.png'  # im6
        l = cv2.imread(path_to_left_rgb_image_file)
        r = cv2.imread(path_to_right_rgb_image_file)

        matcher = CoExMatcher()
        matcher.inference(l, r)

    if VIDEO:
        video_path_l = '../video/red4/left/left.mp4'
        video_path_r = '../video/red4/right/right.mp4'
        cam1 = cv2.VideoCapture(video_path_l)
        cam2 = cv2.VideoCapture(video_path_r)

        frames_total = cam1.get(cv2.CAP_PROP_FRAME_COUNT)
        print("frames_total: ", frames_total)

        result_video_folder = '../video'
        fps = cam1.get(cv2.CAP_PROP_FPS)
        size = (int(cam1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam1.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if RECORDING:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            dst_video = cv2.VideoWriter(result_video_folder + "/output.mp4", fourcc, fps, size, True)

        matcher = CoExMatcher()
        cnt = 0
        _fps = 0.0
        t1 = time.time()
        while True:
            ret1, img1 = cam1.read()
            ret2, img2 = cam2.read()
            cnt += 1

            if not ret1 and ret2:
                print("Done!")
                break
            if img1 is None or img2 is None:
                print("Done!")
                break

            print(cnt)
            _fps = int(cnt / (time.time() - t1))
            print("FPS: %.1f" % _fps)

            matcher.inference(img1, img2)
            if RECORDING:
                dis = cv2.cvtColor(dis, cv2.COLOR_GRAY2BGR)
                dst_video.write(dis)

        print("even FPS: %.1f" % (cnt / (time.time() - t1)))

        dst_video.release()
