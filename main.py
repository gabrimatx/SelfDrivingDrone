import cv2
from djitellopy import Tello
from controller import Controller
from time import sleep

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

controller = Controller(tello, *Controller.SSD_LITE)

while True:
    controller.update()

    if controller.frame_to_stream is None:
        continue
    cv2.imshow("Tello Camera", controller.frame_to_stream)

    if controller.land_signal_detected or cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.end()
cv2.destroyAllWindows()   
    
       