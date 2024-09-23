import argparse
import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseCam:
    def __init__(self, width=640, height=480, fps=30, enable_depth=True):
        """ Initialize the RealSense camera manager. """
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None

        self.depth_enabled = enable_depth

        self.allowed_resolutions = [(1280, 800),
                                    (1280, 720),
                                    (640, 480),
                                    (640, 360),
                                    (480, 270),
                                    (424, 240)]

    def _create_pipeline(self):
        """ Create RealSense pipeline """

        if (self.width, self.height) not in self.allowed_resolutions:
            raise ValueError(f"Invalid resolution: {self.width}:{self.height}\nAllowed resolutions: {self.allowed_resolutions}")

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        if self.depth_enabled:
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)

    def start(self):
        """ Open the camera. """
        if not self._is_device_available():
            raise ValueError("No RealSense device connected.")

        self._create_pipeline()
        # Start streaming
        profile = self.pipeline.start(self.config)

        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.cam_mtx = np.array([[intrinsics.fx, 0.0, intrinsics.ppx],[0.0, intrinsics.fy, intrinsics.ppy],[0.0, 0.0, 1.0]])
        self.cam_dist = np.array(intrinsics.coeffs).reshape(1,5)

        if self.depth_enabled:

            # Setup the 'High Accuracy'-mode
            depth_sensor = profile.get_device().first_depth_sensor()
            preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            for i in range(int(preset_range.max)):
                visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
                print('%02d: %s'%(i,visulpreset))
                if visulpreset == "High Accuracy":
                    depth_sensor.set_option(rs.option.visual_preset, i)

            # enable higher laser-power for better detection
            depth_sensor.set_option(rs.option.laser_power, 180)

            # lower the depth unit for better accuracy and shorter distance covered
            depth_sensor.set_option(rs.option.depth_units, 0.005)

            # Getting the depth sensor's depth scale (see rs-align example for explanation)
            self.depth_scale = depth_sensor.get_depth_scale()
            print("Depth Scale is: " , self.depth_scale)

            # Create an align object
            # rs.align allows us to perform alignment of depth frames to others frames
            # The "align_to" is the stream type to which we plan to align depth frames.
            self.align = rs.align(rs.stream.color)

    def stop(self):
        """ Close the camera. """
        if self.pipeline:
            self.pipeline.stop()
        self.pipeline = None

    def read_frame(self):
        """ Read a frame from the camera. """
        frames = self.pipeline.wait_for_frames()

        if self.depth_enabled:
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                return False, None, None

            
            color_image = np.asanyarray(color_frame.get_data())

            return True, color_image, depth_frame

        else:
            color_frame = frames.get_color_frame()
            if not color_frame:
                return False, None
            
            color_image = np.asanyarray(color_frame.get_data())
            return True, color_image
    
    def get_depth_colormap(self, depth_frame):
        return cv2.applyColorMap(cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03), cv2.COLORMAP_JET)
    
        
    def is_opened(self):
        """ Check if the camera is open. """
        return self.pipeline is not None

    def _is_device_available(self):
        # """ Check if a RealSense device is available. """
        devices = rs.context().query_devices()
        if len(devices) == 0:
            return False
        else:
            try:
                device_name = devices.front().get_info(rs.camera_info.name)
                print("RealSense device found:", device_name)
                return True
            except:
                return False
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=640, help='Width of the frames in the video stream.')
    parser.add_argument('--height', type=int, default=480, help='Height of the frames in the video stream.')
    parser.add_argument('--depth', action='store_true', help='Enable depth stream.')\
    
    args = parser.parse_args()

    cam = RealSenseCam(width=args.width, height=args.height, enable_depth=args.depth)
    cam.start()

    cv2.namedWindow('Realsense Camera', cv2.WINDOW_NORMAL)

    while cam.is_opened():
        if args.depth:
            success, color_image, depth_frame = cam.read_frame()
        else:
            success, color_image = cam.read_frame()
        
        if not success:
            continue

        if args.depth:
            depth_colormap = cam.get_depth_colormap(depth_frame)
            disp_img = np.hstack((color_image, depth_colormap))
        else:
            disp_img = color_image
        
        cv2.imshow('Realsense Camera', disp_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cam.stop()
    cv2.destroyAllWindows()