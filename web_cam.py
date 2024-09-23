import cv2

class WebCam:

    def __init__(self, width=640, height=480, fps=30):
        """ Initialize the OpenCV camera manager. """
        self.width = width
        self.height = height
        self.fps = fps
        self.device = None
        self.depth_enabled = False # for consistency

    def start(self):
        """ Open the camera. """
        self.device = cv2.VideoCapture(0)
        self.device.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.device.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.device.set(cv2.CAP_PROP_FPS, self.fps)
    
    def is_opened(self):
        """ Check if the camera is opened. """
        return self.device.isOpened()

    def stop(self):
        """ Close the camera. """
        self.device.release()
        self.device = None

    def read_frame(self):
        """ Read a frame from the camera. """
        ret, frame = self.device.read()
        return ret, frame