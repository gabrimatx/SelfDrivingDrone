import numpy as np
import cv2
from djitellopy import Tello
from simple_pid import PID
from obstacles_detector import ObstaclesDetector
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Controller:
    """
    Class for drone autonomous control.
    """
    CASCADE = "models/cascade.xml", "Cascade"
    FASTER_RCNN = "models/fasterrcnn.pth", "Faster R-CNN"
    SSD_LITE = "models/ssdlite.pth", "SSDLite"
    
    def __init__(self, tello: Tello, model_path: str, model_type: str = "SDDLite") -> None:
        self.tello = tello
        self.frame_to_stream = None

        self.obstacles_detector = ObstaclesDetector(model_path, model_type)

        self.frame_width = 360
        self.frame_height = 240

        self.pid_x = PID(-0.3, -0.01, -0.1, setpoint=self.frame_width // 2)
        self.pid_y = PID(0.3, 0.01, 0.1, setpoint=self.frame_height // 2 - 30)
        self.pid_d = PID(0.0003, 0, 0, setpoint=self.frame_height*self.frame_width)

        self._initialize_plot()

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
        if len(boxes) == 0:
            self.tello.send_rc_control(0,0,0,0)
            return

        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
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

    def _initialize_plot(self):
        """
        Initializer of the subplots for the real-time plot.
        """
        self.error_history_x = []
        self.error_history_y = []
        self.error_history_d = []
        self.fig, axs = plt.subplots(3, 1, figsize=(20,18))
        self.fig.subplots_adjust(hspace=0.5)
        self.fig.set_dpi(70)

        x = np.linspace(0,1000,1000)
        y = np.zeros((1000))
        variables = ["x", "y", "d"]
        for i in range(3):
            axs[i].set_xlim(0, 100)
            axs[i].set_ylim(-200, 200)
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel(f'Error on {variables[i]}')
            axs[i].set_title(f'Real-Time Error Plot for {variables[i]}')
            axs[i].plot(x,  y, linestyle='--', color='grey', label='Dotted Line')

        self.line0, = axs[0].plot([], [], lw=2, color='red')
        self.line1, = axs[1].plot([], [], lw=2, color='blue')
        self.line2, = axs[2].plot([], [], lw=2, color='green')

        self.ani = FuncAnimation(self.fig, self._ani_func, init_func=self._ani_init_func, frames=10000, interval=500, blit=True)
        plt.show()


    # initialization function: plot the background of each frame
    def _ani_init_func(self):
        """
        Initializer for the FuncAnimation object.
        """
        self.line0.set_data([], [])
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        return self.line0, self.line1, self.line2,

    def _ani_func(self, i):
        """
        Function called each frame by FuncAnimation object.
        """
        error_x = self.pid_x._last_error if self.pid_x._last_error else 0
        error_y = self.pid_y._last_error if self.pid_y._last_error else 0
        error_d = self.pid_d._last_error if self.pid_d._last_error else 0

        ms=i/2
        self.times.append(ms)
        self.error_history_x.append(error_x)
        self.error_history_y.append(error_y)
        self.error_history_d.append(error_d)

        x0 = self.times
        y0 = self.error_history_x
        self.line0.set_data(x0, y0)

        x1 = self.times
        y1 = self.error_history_y
        self.line1.set_data(x1, y1)

        x2 = self.times
        y2 = self.error_history_d
        self.line2.set_data(x2, y2)

        return self.line0, self.line1, self.line2,