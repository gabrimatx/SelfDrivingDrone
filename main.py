import cv2
import sys
from djitellopy import Tello
from pid_based_controller import Controller
from time import sleep

default_model = "ssdlite"
argument_dict = {"cascade":Controller.CASCADE, "ssdlite":Controller.SSD_LITE, "faster-rcnn": Controller.FASTER_RCNN}
try:
    modality = argument_dict[sys.argv[1].lower()]
except:
    modality = argument_dict[default_model]

print(f"SELF DRIVING DRONE")
print(f"Selected model: {modality[1]}")

print("Connecting to drone...")
tello = Tello()
tello.connect()
tello.streamon()
print("Done")
print(f"Tello battery: {tello.get_battery()}%")

print("Loading model and controller...")
controller = Controller(tello, *modality)
print("Done")

print("Loading Tello camera...")
while tello.get_frame_read().frame is None:
    continue
print("Done")

print("Tello ready to takeoff")
tello.send_rc_control(0,0,0,0)
tello.takeoff()


video_writer_drone = cv2.VideoWriter("video_drone.mp4", 
                         cv2.VideoWriter_fourcc(*"MP4V"),
                         10, (360, 240))
video_writer_plot = cv2.VideoWriter("video_errors.mp4", 
                                      cv2.VideoWriter_fourcc(*"MP4V"),
                                      10, (1170, 900))

while True:
    controller.update()

    if controller.frame_to_stream is None:
        continue

    video_writer_drone.write(controller.frame_to_stream)
    video_writer_plot.write(controller.plot_to_stream)

    cv2.imshow("Tello Camera", controller.frame_to_stream)
    cv2.imshow("Errors", controller.plot_to_stream)
    cv2.moveWindow("Errors", 0, controller.output_frame_height - 50)

    if not(controller._land) and cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.end()
cv2.destroyAllWindows()   
