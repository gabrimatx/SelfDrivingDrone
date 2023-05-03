from threading import Thread
import cv2
from djitellopy import Tello
from controller import Controller

tello = Tello()

tello.connect()
tello.streamon()
tello.takeoff()

controller = Controller(tello)

def stream() -> None:
    """
    Streams processed frames, called by streamer thread.
    """
    while controller.processed_frame is not None:
        cv2.imshow("Tello Camera", controller.processed_frame)

streamer = Thread(target = stream, daemon=True)
streamer.start()

while not controller.land_signal_detected:
    controller.update()

tello.land()
    
       