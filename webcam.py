import cv2
from waiting import wait
from flask import Flask, render_template, Response

frames = []

#def is_something_ready(something):
#    if something:
 #       return True
 #   return False

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    frames.append(frame)
    cv2.imshow("Webcam", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

'''x = 1
for i in range(len(frames)):
    cv2.imshow("frames", frames[i])
    print(x)
    x += 1
    cv2.waitKey(20)'''
    
'''cv2.waitKey(0)



#vc.release()
#cv2.destroyWindow("preview")'''