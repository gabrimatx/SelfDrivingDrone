import cv2

cap = cv2.VideoCapture('neg.mov')
pos_text = open("neg.txt", "w")

counter = 0
iter = 0
while True:
    ret, frame = cap.read()

    if frame is None:
        break

    if iter % 2 == 0:
        cv2.imwrite(f"neg/neg{counter}.jpg", frame)
        pos_text.write(f"neg/neg{counter}.jpg\n")
        counter += 1

    iter += 1

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
pos_text.close()
cv2.destroyAllWindows()