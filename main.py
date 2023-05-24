from threading import Thread
import cv2
from djitellopy import Tello
from controller import Controller

tello = Tello()

tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()
tello.send_rc_control(0,0,0,0)
#tello.takeoff()
#tello.move_up(70)

controller = Controller(tello)

while True:
    controller.update()

    if controller.frame_to_stream is None:
        continue
    cv2.imshow("Tello Camera", controller.frame_to_stream)

    if controller.land_signal_detected or cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.end()
cv2.destroyAllWindows()
    
       