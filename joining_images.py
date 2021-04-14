import cv2
import numpy as np


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[00][0].shape[0]
    if rowsAvailable:
        for x in range(0,rows):
            for y in range(0,cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0),None,scale,scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y],(imgArray[0][0].shape[1],imgArray[0][0].shape[0]))
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y],imgArray[0][0].shape[2])
        imgBlank = np.zeros((height,width,3),np.uint8)
        hor = [imgBlank]*rows
        hor_con = [imgBlank]*rows
        for x in range(0,rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0,rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]))
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], imgArray[0].shape[2])
        hor = np.hstack(imgArray)
        ver = hor
    return  ver


#img = cv2.imread("resources/el5n9i.jpg")
#imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#imgStack = stackImages(0.5,([img,imgGray,img],[img,img,img]))

#cv2.imshow("vertical",img)
#cv2.imshow("Stackimages",imgStack)

#cv2.waitKey(0)
