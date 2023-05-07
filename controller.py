import numpy as np
import cv2
from djitellopy import Tello

class Controller:
    """
    Autonomous controller for the drone.
    """
    def __init__(self, tello: Tello) -> None:
        self.tello = tello

        self.processed_frame = None

        self.land_signal_detected = False

    def update(self) -> None:
        """
        Updates the position of the drone in the environment. 
        """
        obstacle_position = self._detect_nearest_obstacle()
        self._compute_and_send_controls(obstacle_position)

    def _detect_nearest_obstacle(self) -> np.ndarray:
        """
        Detects an obstacles and returns its position in the frame.
        """
        frame = self.tello.get_frame_read().frame
        pass

    def _draw_on_frame(self, frame, to_draw) -> None:
        """
        Takes as an input a raw frame and draws on it.
        """
        pass

    def _compute_and_send_controls(self, obstacle_position) -> None:
        """
        Computes controls according to obstacle position and sends them to the drone.
        """
        pass