#Importing Modules
import cv2
import numpy as np
from joining_images import stackImages


def getContoours(img, imgContour):
    contours, hierarcy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    print(f"Number of counters:{len(contours)}")
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(f'\narea={area}')
        if area > 0:

            peri = cv2.arcLength(cnt, True)
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if objCor == 3: objectType = "Tri"
            elif objCor == 4:
                aspratio = w/float(h)
                if aspratio > 0.95 and aspratio < 1.05: objectType ="Square"
                else: objectType = "Rect"
            elif objCor == 8:
                objectType = "Circle"
            elif objCor == 6:
                objectType = "Hex"
            else:
                objectType = None

            #cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 155, 0), 2)
            cv2.putText(imgContour,f"{objCor}{objectType}",
                        (x + int(w / 2) - 10, y + int(h / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)
#main
path = "resources/shapes_small.jpg"
img = cv2.imread(path)
imgContour = img.copy()
imgContour2 = img.copy()
imgBlur = cv2.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
imgthresh = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,11,2)

#Result
#cv2.resize(imgthresh,(img.shape[1],img.shape[0]))
imgCanny = cv2.Canny(imgBlur, 50, 50)
getContoours(imgCanny,imgContour)
getContoours(imgthresh,imgContour2)
imgBlank = np.zeros_like(img)
imgStack = stackImages(0.6, ([img, imgthresh, imgGray],
                             [imgCanny, imgContour, imgContour2]))

cv2.imshow("Stack", imgStack)
cv2.waitKey(0)
