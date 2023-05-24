import numpy as np
import cv2
from djitellopy import Tello
from simple_pid import PID

class Controller:
    """
    Autonomous controller for the drone.
    """
    def __init__(self, tello: Tello) -> None:
        self.tello = tello
        self.frame_to_stream = None
        self.land_signal_detected = False
        self.cascade_classifier = cv2.CascadeClassifier("cascade/cascade.xml")

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
        obstacles = self._detect_obstacles(frame)
        self._draw_box_on_obstacles(frame, obstacles)
        self.frame_to_stream = frame
        self._compute_and_send_controls(obstacles)
        
    def _detect_obstacles(self, frame: np.ndarray) -> np.ndarray:
        """
        Detects an obstacles and returns its position in the frame.
        """
        processed_frame = self._preprocess(frame)
        obstacles = self.cascade_classifier.detectMultiScale(processed_frame, 1.35, 30)
        return obstacles
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies filters to the frame before passing it to the Cascade Classifier.
        """
        processed_frame = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))[0]
        processed_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        return processed_frame

    def _draw_box_on_obstacles(self, frame: np.ndarray, obstacles:np.ndarray) -> None:
        """
        Takes as an input a raw frame and draws a box on every detected obstacle.
        """
        for obstacle in obstacles:
            up_left = obstacle[:2]
            down_right = up_left + obstacle[2:]
            cv2.rectangle(frame, up_left, down_right, (0, 0, 255), 2)

            center = (up_left + down_right) // 2
            cv2.circle(frame, center, 5, (0, 255, 0), cv2.FILLED)

    def _compute_and_send_controls(self, obstacles:np.ndarray) -> None:
        """
        Computes controls according to obstacle position and sends them to the drone.
        This is achieved using a PID control.
        """
        if len(obstacles) == 0:
            self.tello.send_rc_control(0, 0, 0, 0)
            return
        
        max_area_i = np.argmax(np.prod(obstacles[:, 2:], axis=1))
        area = np.prod(obstacles[max_area_i, 2:])
        center = obstacles[max_area_i, :2]
        
        left_right_velocity = int(self.pid_x(center[0]))
        up_down_velocity = int(self.pid_y(center[1]))
        forward_backward_velocity = int(self.pid_d(area))

        self.tello.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, 0)