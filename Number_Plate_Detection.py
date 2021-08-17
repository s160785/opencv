#Importing Modules
import cv2
import numpy as np
import pytesseract   #Optimal Character Recognition tool

##################################################

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
frameWidth = 640
frameHeight = 480
count = 0

#Using haarcascade xml file to detect number plate in an image
numberPlateCascade = cv2.CascadeClassifier("resources/haarcascades/haarcascade_russian_plate_number.xml")
###################################################
#video capture 
cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,130) # Brightness

#main
while True:
    success,img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    numberPlates = numberPlateCascade.detectMultiScale(imgGray, 1.1, 6)
    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > 500:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img,"Number Plate ",(x,y-5),
                        cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
            imgRegionOfInterest = img[y:y+h,x:x+w]
            cv2.imshow("ROI",imgRegionOfInterest)
    #Result
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("resources/Number_plates/number_plate_"+str(count)+".jpg",imgRegionOfInterest)
        cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,"Scan Saved",(150,265),cv2.FONT_HERSHEY_DUPLEX,
                    2,(0,0,255),2)
        cv2.imshow("Result",img)
        cv2.waitKey(500)
        count += 1
