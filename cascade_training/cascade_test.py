import cv2
import numpy as np
from djitellopy import Tello

def detect_nearest_target(img):
    cascade = cv2.CascadeClassifier("cascade/cascade.xml")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    targets = cascade.detectMultiScale(img_gray, 1.1, 60)
    if len(targets) == 0:
        return None, None
    
    max_area = 0
    max_area_center = None    
    for target in targets:
        up_left = target[:2]
        down_right = up_left + target[2:]
        cv2.rectangle(img, up_left, down_right, (0, 0, 255), 2)

        area = np.prod(target[2:])
        center = (up_left + down_right) // 2
        cv2.circle(img, center, 5, (0, 255, 0), cv2.FILLED)

        if area > max_area:
            max_area = area
            max_area_center = center
        
    return max_area, max_area_center

def detect_contours(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5,5,), 0)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse= True)
    cv2.drawContours(img, contours, -1, (0,255,0), 3)

cap = cv2.VideoCapture('test_video.mov')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    detect_nearest_target(frame)
    # Display the resulting frame
    cv2.imshow(f"test", frame)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()