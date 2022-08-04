import cv2
import numpy as np
import time


frame_count = 0
cap=cv2.VideoCapture("C:\\Users\\anand\\Documents\\UTAR\\Y3S1\\FYP Project\\Code\\VideoSource\\video2.mp4")

while(cap.isOpened()):
    ret,frame=cap.read()
    frame=cv2.resize(frame,(640,480))

    if ret == True:
        frame_count += 1

        if(frame_count % 25) == 0: 
            cv2.imwrite('C:\\Users\\anand\\Documents\\UTAR\Y3S1\\FYP Project\\Code\\BackgroundSubtractionAnand\\vehicledetechtion{}.jpg'.format(frame_count), frame) 

        if (frame_count >= 1500):
            cap.release()
            cv2.destroyAllWindows()

        if cv2.waitKey(1)&0xff==ord('q'):
            break



    else:
        break

cap.release()
cv2.destroyAllWindows()
