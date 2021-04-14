import cv2
import numpy as np
from joining_images import stackImages
def empty(a):
    pass


frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,130) # Brightness

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 240)
cv2.createTrackbar("Hue Min", "Trackbars", 10, 179,empty)
cv2.createTrackbar("Saturation Min", "Trackbars", 0, 255,empty)
cv2.createTrackbar("Value Min", "Trackbars", 45, 255, empty)
cv2.createTrackbar("Hue Max", "Trackbars", 179, 179,empty)
cv2.createTrackbar("Saturation Max", "Trackbars", 63, 255,empty)
cv2.createTrackbar("Value Max", "Trackbars", 213, 255,empty)

while True:
    success, img = cap.read()


    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hue_min = cv2.getTrackbarPos("Hue Min","Trackbars")
    hue_max = cv2.getTrackbarPos("Hue Max","Trackbars")
    sat_min = cv2.getTrackbarPos("Saturation Min","Trackbars")
    sat_max = cv2.getTrackbarPos("Saturation Max","Trackbars")
    val_min = cv2.getTrackbarPos("Value Min","Trackbars")
    val_max = cv2.getTrackbarPos("Value Max","Trackbars")
    print(hue_min,hue_max,sat_min,sat_max,val_min,val_max)
    lower = np.array([hue_min,sat_min,val_min])
    upper =np.array([hue_max,sat_max,val_max])
    mask =cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img, mask=mask)

    #cv2.imshow("original",img)
    #cv2.imshow("HSV",imgHSV)
    #cv2.imshow("Mask",mask)
    #cv2.imshow("Result", imgResult)

    imgStack = stackImages(0.6,([img,imgHSV],[mask,imgResult]))

    cv2.imshow("Stacked",imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break