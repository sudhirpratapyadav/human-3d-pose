#!/usr/bin/env python

import argparse
import math
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe import solutions
import time
from scipy.spatial.transform import Rotation as R

from web_cam import WebCam
from realsense_cam import RealSenseCam

from pose_visualiser import PoseVisualiser
from timing import TimingLogger


def is_valid(value: float) -> bool:
    return 0 <= value <= 1 or math.isclose(value, 0) or math.isclose(value, 1)

class PoseTracker:
    def __init__(self, cam_type='webcam', width=640, height=480, enable_depth=False, model_name='lite'):
        
        model_asset_path = os.path.join(os.path.dirname(__file__), f'models/pose_landmarker_{model_name}.task')
        self.options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_asset_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO)
        
        self.cam = self._initialize_camera(cam_type, width, height, enable_depth)
        if enable_depth:
            self.depth_range = (0.1, 4.0)

        self._PRESENCE_THRESHOLD = 0.5
        self._VISIBILITY_THRESHOLD = 0.5

        self.tranform_pcd = np.dot(R.from_euler('x', 90, degrees=True).as_matrix(), R.from_euler('z', 180, degrees=True).as_matrix())

        cam_height = 1.2 # height of camera from ground
        self.cam2world = np.array([
                            [0, 0, 1, 0],
                            [-1, 0, 0, 0],
                            [0, -1, 0, cam_height],
                            [0, 0, 0, 1]])


    def _initialize_camera(self, cam_type, width, height, enable_depth):
        """ Initialize the camera based on the specified type. """
        if cam_type == 'realsense':
            cam = RealSenseCam(enable_depth=enable_depth, width=width, height=height)
            return cam
        else:
            return WebCam(width=width, height=height)
    
    def draw_landmarks_on_image(self, color_img, depth_frame, keypoints_2d):
        circle_radius = 2
        circle_border_radius = max(circle_radius + 1, int(circle_radius * 1.2))
        thickness = 1
        if depth_frame is not None:
            depth_image = self.cam.get_depth_colormap(depth_frame)
        for connection in solutions.pose.POSE_CONNECTIONS:
            start = keypoints_2d[connection[0]]
            end = keypoints_2d[connection[1]]
            if not (start is None or end is None):
                cv2.line(color_img, start, end, (255, 255, 255), thickness)
                if depth_frame is not None:
                    cv2.line(depth_image, start, end, (255, 255, 255), thickness)
                    
        for point in keypoints_2d:
            if point is not None:
                cv2.circle(color_img, point, circle_border_radius, (255, 255, 255), thickness)
                cv2.circle(color_img, point, circle_radius, (0, 255, 0), thickness)
                if depth_frame is not None: 
                    cv2.circle(depth_image, point, circle_border_radius, (255, 255, 255), thickness)
                    cv2.circle(depth_image, point, circle_radius, (0, 0, 255), thickness)
                    
        
        color_img = cv2.flip(color_img, 1)
        if depth_frame is not None:
            depth_image = cv2.flip(depth_image, 1)
            disp_img = np.hstack((color_img, depth_image))
        else:
            disp_img = color_img
        return disp_img
        
    
    def calc_key_points(self, pos_res, depth_frame=None):
        keypoints_3d = []
        keypoints_2d = []
        if depth_frame is None:
            for landmark_2d, landmark_3d in zip(pos_res.pose_landmarks[0], pos_res.pose_world_landmarks[0]):
                if landmark_2d.visibility > self._VISIBILITY_THRESHOLD and landmark_2d.presence > self._PRESENCE_THRESHOLD and is_valid(landmark_2d.x) and is_valid(landmark_2d.y):
                    px = min(math.floor(landmark_2d.x * self.cam.width), self.cam.width - 1)
                    py = min(math.floor(landmark_2d.y * self.cam.height), self.cam.height - 1)
                    keypoints_2d.append([px, py])
                    keypoints_3d.append([landmark_3d.x, landmark_3d.y, landmark_3d.z])
                else:
                    keypoints_3d.append([np.nan, np.nan, np.nan])
                    keypoints_2d.append(None)
        else:
            for landmark in pos_res.pose_landmarks[0]:
                if landmark.visibility > self._VISIBILITY_THRESHOLD and landmark.presence > self._PRESENCE_THRESHOLD and is_valid(landmark.x) and is_valid(landmark.y):
                    px = min(math.floor(landmark.x * self.cam.width), self.cam.width - 1)
                    py = min(math.floor(landmark.y * self.cam.height), self.cam.height - 1)
                    keypoints_2d.append([px, py])

                    kp_depth = depth_frame.get_distance(px, py)
                    if kp_depth < self.depth_range[0] or kp_depth > self.depth_range[1]:
                        keypoints_3d.append([np.nan, np.nan, np.nan, 1])
                        continue
                    xyz_img = kp_depth*np.array([px,py,1])
                    xyz_c = np.matmul(np.linalg.inv(self.cam.cam_mtx), xyz_img.T)
                    keypoints_3d.append(xyz_c.reshape(3).tolist()+[1])
                else:
                    keypoints_3d.append([np.nan, np.nan, np.nan, 1])
                    keypoints_2d.append(None)
        keypoints_3d = np.array(keypoints_3d)
        if depth_frame is None:
            keypoints_3d = keypoints_3d.dot(self.tranform_pcd)
        else:
            keypoints_3d = np.dot(keypoints_3d, self.cam2world.T)[:, :3]
        return keypoints_2d, keypoints_3d
    
    def start_tracking(self):
        """ Start the hand tracking process using the selected camera. """
        self.cam.start()

        cv2.namedWindow('2D Pose', cv2.WINDOW_NORMAL)

        vis = PoseVisualiser(connections=np.array(list(solutions.pose.POSE_CONNECTIONS)),
                             show_grid=True,
                             camera_pose=self.cam2world,
                             cam_fov_v=65,
                             img_width=self.cam.width,
                             img_height=self.cam.height,
                             display_image=True,
                             camera_intrinsics=self.cam.cam_mtx)

        time_log = TimingLogger()
        try:
            with mp.tasks.vision.PoseLandmarker.create_from_options(self.options) as pose_model:
                while self.cam.is_opened():
                    time_log.next()
                    if self.cam.depth_enabled:
                        success, color_image, depth_frame = self.cam.read_frame()
                    else:
                        success, color_image = self.cam.read_frame()
                        depth_frame = None
                    if not success:
                        continue
                    time_log.stamp('read_frame')
                    
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)
                    pos_res = pose_model.detect_for_video(mp_image, int(time.time() * 1000))
                    time_log.stamp('pose_detect')

                    if pos_res.pose_landmarks:
                        key_points_2d, key_points_3d = self.calc_key_points(pos_res, depth_frame)
                        disp_img = self.draw_landmarks_on_image(color_image, depth_frame, key_points_2d)

                        vis.update(points=key_points_3d, frame=cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

                    else:
                        if self.cam.depth_enabled:
                            disp_img = np.hstack((cv2.flip(color_image, 1), cv2.flip(self.cam.get_depth_colormap(depth_frame), 1)))
                        else:
                            disp_img = cv2.flip(color_image, 1)
                    time_log.stamp('keypoints')

                    fps_text = f'FPS: {time_log.fps:.1f}'
                    cv2.putText(disp_img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('2D Pose', disp_img)

                    time_log.stamp('display')

                    # Exit on 'ESC' key
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            time_log.report(ignore_init_iters=3, include_iters=False)
        finally:
            self.cam.stop()
            cv2.destroyAllWindows()
            vis.close()

def main():
    # take argmuent for depth
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam-type', type=str, default='realsense', help='type of camera {realsense, webcam}')
    parser.add_argument('--depth', action='store_true', help='Enable depth (in-case of depth camera)')
    parser.add_argument('--model', type=str, default='lite', help='Model name {lite, full, heavy}')

    parser.add_argument('--width', type=int, default=640, help='Width of the Image')
    parser.add_argument('--height', type=int, default=480, help='Height of the Image')
    args = parser.parse_args()

    pose_tracker = PoseTracker(cam_type=args.cam_type, enable_depth=args.depth, model_name=args.model)
    pose_tracker.start_tracking()

if __name__ == "__main__":
    main()
