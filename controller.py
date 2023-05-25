import numpy as np
import cv2
from djitellopy import Tello
from simple_pid import PID
from obstacles_detector import ObstaclesDetector

class Controller:
    """
    Class for drone autonomous control.
    """
    CASCADE = "models/cascade.xml", "Cascade"
    FASTER_RCNN = "models/fasterrcnn.pth", "Faster R-CNN"
    SSD_LITE = "models/sddlite.pth", "SSDLite"
    def __init__(self, tello: Tello, model_path: str, model_type: str = "SDDLite") -> None:
        self.tello = tello
        self.frame_to_stream = None
        self.land_signal_detected = False

        self.obstacles_detector = ObstaclesDetector(model_path, model_type)

        self.frame_width = 360
        self.frame_height = 240

        self.pid_x = PID(-0.2, -0.01, -0.1, setpoint=self.frame_width // 2)
        self.pid_y = PID(0.2, 0.01, 0.1, setpoint=self.frame_height // 2 - 30)
        self.pid_d = PID(0.0006, 0, 0.0003, setpoint=self.frame_height*self.frame_width)

    def update(self) -> None:
        """
        Updates the position of the drone in the environment and the frame to stream. 
        """
        frame = cv2.resize(self.tello.get_frame_read().frame, (self.frame_width, self.frame_height))

        if frame is None:
            return

        boxes = self.obstacles_detector.detect_obstacles(frame)
        self.obstacles_detector.draw_bounding_boxes(frame, boxes)

        self.frame_to_stream = frame
        self._compute_and_send_controls(boxes)

    def _compute_and_send_controls(self, boxes:np.ndarray):
        """
        Computes controls according to obstacle position and sends them to the drone.
        This is achieved using a PID control.
        """

        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        if len(areas) == 0:
            return
        
        max_area_i = np.argmax(areas)
        max_area = areas[max_area_i]
        max_area_box = boxes[max_area_i]
        
        up_left = max_area_box[:2]
        down_right = max_area_box[2:]
        center = (up_left + down_right) // 2

        left_right_velocity = int(self.pid_x(center[0]))
        up_down_velocity = int(self.pid_y(center[1]))
        forward_backward_velocity = int(self.pid_d(max_area))

        self.tello.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, 0)