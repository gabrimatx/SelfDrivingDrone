import cv2
import shutil
import os

i, j = 0, 0
basestr = 'pos' 
videoname = "pos.mp4"
basepath = [basestr + y for y in ('G', 'L', 'R')]
infotxts = ['', '', '']

vidcap = cv2.VideoCapture(videoname)
success,image = vidcap.read()

# [shutil.rmtree(basepath[i]) for i in range(3)]
[os.makedirs(basepath[i]) for i in range(3)]

while success:
  if not i % 10:
    cv2.imwrite(f"{basepath[j]}/{i//10}.jpg", image)     # save frame as JPEG file 
    print(f'Read a frame {i}, it went to {basepath[j]}: ', success)
    infotxts[j] += f"{basepath[j]}/{i//10}.jpg\n"
    j+=1
    if j==3:
      j=0
  success,image = vidcap.read()
  i += 1

for i in range(3):
  with open(f"{basepath[i]}/info.txt", 'w') as fh:
    fh.write(infotxts[i][:-1])