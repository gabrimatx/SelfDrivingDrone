import numpy as np
import cv2
from djitellopy import Tello
from simple_pid import PID
from obstacles_detector import ObstaclesDetector
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

class Controller:
    """
    Class for drone autonomous control.
    """
    CASCADE = "models/cascade.xml", "Cascade"
    FASTER_RCNN = "models/fasterrcnn.pth", "Faster R-CNN"
    SSD_LITE = "models/ssdlite.pth", "SSDLite"
    
    def __init__(self, tello: Tello, model_path: str = "models/ssdlite.pth", model_type: str = "SSDLite") -> None:
        self._tello = tello
        self.frame_to_stream = None
        self.plot_to_stream = None
        self._controller_off = False
        self._land = False
        self._zero_boxes_counter = 0

        self._obstacles_detector = ObstaclesDetector(model_path, model_type)

        self.output_frame_width = 360
        self.output_frame_height = 240

        self._pid_x = PID(-0.24, -0.013, -0.067, setpoint=self.output_frame_width // 2)
        self._pid_y = PID(0.3, 0.002, 0.1, setpoint=self.output_frame_height // 2 - 70)
        self._pid_d = PID(0.0005, 0.00003, 0.000016, setpoint=self.output_frame_height*self.output_frame_width // 4)

        self._initialize_plot()

    def update(self) -> None:
        """
        Updates the position of the drone in the environment and the frame to stream. 
        """
        frame = cv2.resize(self._tello.get_frame_read().frame, (self.output_frame_width, self.output_frame_height))

        if frame is None:
            return

        boxes = self._obstacles_detector.detect_obstacles(frame)
        self._obstacles_detector.draw_bounding_boxes(frame, boxes)
        self.frame_to_stream = frame
        self.plot_to_stream = self._compute_and_send_controls(boxes)
        
    def _compute_and_send_controls(self, boxes: np.ndarray):
        """
        Computes controls according to obstacle position and sends them to the drone.
        This is achieved using a PID control.
        """
        if self._controller_off:
            if time.time() - self._start_passing_through > 2.5:
                self._controller_off = False
            plot_img = self._update_and_get_plot_img()
            return plot_img

        if self._zero_boxes_counter == 20:
            # When no box is detected for more than 120 times,
            # try moving up (obstacle could be out of vision range)
            self._tello.send_rc_control(0, 0, 40, 0)
            self._start_passing_through = time.time()
            self._controller_off = True
        
        elif self._zero_boxes_counter == 100:
            # When no box is detected for more than 160 times,
            # land (it is likely that there are no more obstacles)
            self._land = True

        if len(boxes) == 0:
            self._zero_boxes_counter += 1
            self._tello.send_rc_control(0, 0, 0, 0)
        
        else:
            self._zero_boxes_counter = 0
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            
            max_area_i = np.argmax(areas)
            max_area = areas[max_area_i]
            max_area_box = boxes[max_area_i]
            
            up_left = max_area_box[:2]
            down_right = max_area_box[2:]
            center = (up_left + down_right) // 2
            error_xy = abs(center - np.array([self._pid_x.setpoint, self._pid_y.setpoint]))
            error_area = abs(max_area - self._pid_d.setpoint)

            if (error_xy < 25).all():
                if error_area < 2000:
                    # Tello is ready to pass through the obstacle
                    self._tello.send_rc_control(0, 30, 0, 0)
                    self._start_passing_through = time.time()
                    self._controller_off = True
                    plot_img = self._update_and_get_plot_img()
                    return plot_img
                
                else:
                    forward_backward_velocity = int(self._pid_d(max_area))
            else:
                forward_backward_velocity = 0

            left_right_velocity = int(self._pid_x(center[0]))
            up_down_velocity = int(self._pid_y(center[1]))
            self._tello.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, 0)

        plot_img = self._update_and_get_plot_img()
        return plot_img

    def _initialize_plot(self):
        """
        Initializes the subplots for the real-time errors plot.
        """
        self._error_history_x = []
        self._error_history_y = []
        self._error_history_d = []

        self._start_time = time.time()
        self._times = []

        self._fig, self._axs = plt.subplots(3, 1, figsize=(13,10))
        self._fig.subplots_adjust(hspace=0.5)
        self._fig.set_dpi(90)

        variables = ["x", "y", "d"]
        for i in range(3):
            self._axs[i].grid()
            self._axs[i].set_xlabel('Time (s)')
            self._axs[i].set_ylabel(f'Error on {variables[i]}')
            self._axs[i].set_title(f'Real-Time Error Plot for {variables[i]}')
            
        self._line0, = self._axs[0].plot([], [], lw=2, color='red',)
        self._line1, = self._axs[1].plot([], [], lw=2, color='blue')
        self._line2, = self._axs[2].plot([], [], lw=2, color='green')
        self._axs[0].relim(); self._axs[1].relim(); self._axs[2].relim()
        self._axs[0].autoscale(); self._axs[1].autoscale(); self._axs[2].autoscale()

    def _update_and_get_plot_img(self):
        """
        Updates the real-time errors plot.
        """
        error_x = self._pid_x._last_error if self._pid_x._last_error else 0
        error_y = self._pid_y._last_error if self._pid_y._last_error else 0
        error_d = self._pid_d._last_error if self._pid_d._last_error else 0

        self._times.append(time.time() - self._start_time)
        self._error_history_x.append(error_x)
        self._error_history_y.append(error_y)
        self._error_history_d.append(error_d)

        x0 = self._times
        y0 = self._error_history_x
        self._line0.set_data(x0, y0)

        x1 = self._times
        y1 = self._error_history_y
        self._line1.set_data(x1, y1)

        x2 = self._times
        y2 = self._error_history_d
        self._line2.set_data(x2, y2)

        self._axs[0].relim(); self._axs[1].relim(); self._axs[2].relim()
        self._axs[0].autoscale(); self._axs[1].autoscale(); self._axs[2].autoscale()

        self._fig.canvas.draw()
        img_plot = cv2.cvtColor(np.asarray(self._fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        return img_plot