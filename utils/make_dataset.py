import cv2

cap = cv2.VideoCapture('neg.mov')

counter = 0
iter = 0
while True:
    ret, frame = cap.read()

    if frame is None:
        break

    if iter % 10 == 0:
        cv2.imwrite(f"neg/neg{counter}.jpg", frame)
        counter += 1

    iter += 1

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()