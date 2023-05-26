import cv2
import sys
from djitellopy import Tello
from controller import Controller
from time import sleep
default_model = "ssd-lite"


argument_dict = {"cascade":Controller.CASCADE, "ssd-lite":Controller.SSD_LITE, "faster-rcnn": Controller.FASTER_RCNN}
try:
    modality = argument_dict[sys[1].lower()]
except:
    modality = argument_dict(default_model.lower())

tello = Tello()

tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()

while tello.get_frame_read().frame is None:
    continue

tello.send_rc_control(0,0,0,0)
tello.takeoff()
tello.send_rc_control(0, 0, 50, 0)
sleep(2.5)

controller = Controller(tello, *modality)

while True:
    controller.update()

    if controller.frame_to_stream is None:
        continue
    cv2.imshow("Tello Camera", controller.frame_to_stream)
    cv2.imshow("Errors", controller.plot_to_stream)
    cv2.moveWindow("Errors", 0, controller.output_frame_height - 40)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.end()
cv2.destroyAllWindows()   
