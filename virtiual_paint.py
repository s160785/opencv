import cv2
import numpy as np
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,130) # Brightness
myColors = [[0,106,100,118,255,255], # blue
             [5,107,0,19,255,255], # orange
             [133,56,0,159,156,255], # purple
             [57,76,0,100,255,255]]  # green
myColorValues = [[255,0,0],[51,153,255],
                 [255,0,255],[0,255,0]]  #BGR format
myPoints = [] #x,y,Colorid

def findColor(img,myColors,imgResult,myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x,y = getContoours(mask,imgResult)
        cv2.circle(imgResult,(x,y),10,myColorValues[count],cv2.FILLED)
        if x!=0 and y!=0:
            newPoints.append([x,y,count])
        count += 1
        #cv2.imshow(str(color[0]),mask)
    return newPoints

def getContoours(img, imgContour):
    contours, hierarcy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 0:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #cv2.drawContours(imgContour, cnt, -1, (0, 0, 255), 3)
            x, y, w, h = cv2.boundingRect(approx)
    return int(x+w/2),y

def DrawOnCanvas(myPoints,myColorValues):
    for point in myPoints:
        cv2.circle(imgResult,(point[0],point[1]),10,myColorValues[point[2]],cv2.FILLED)
while True:
    success,img = cap.read()
    imgResult = img.copy()
    newPoints = findColor(img,myColors,imgResult,myColorValues)
    if len(newPoints)!=0:
        for point in newPoints:
            myPoints.append(point)
    if len(myPoints)!=0:
        DrawOnCanvas(myPoints,myColorValues)

    cv2.imshow("Result",imgResult)
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break