#Importing Modules
import cv2
import numpy as np
from joining_images import stackImages

#Using haarcasscade xml files to detect faces in image
faceCascade = cv2.CascadeClassifier("resources/haarcascades/haarcascade_frontalface_default.xml")
img = cv2.imread("resources/multiple_faces.jpg")
img = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

#Converting to Gray Image 
imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

faces = faceCascade.detectMultiScale(imgGray,1.1,6)
#Result
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow("Result", img)
cv2.waitKey(0)
