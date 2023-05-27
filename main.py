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

controller = Controller(tello, *Controller.SSD_LITE)

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