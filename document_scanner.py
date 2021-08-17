#Importing Modules
import cv2
import numpy as np
###################################################################
widthImg = 640
heightImg =480
#####################################################################
#Video Capture
# cap = cv2.VideoCapture(0)
# cap.set(3, widthImg)
# cap.set(4, heightImg)
# cap.set(10,130) # Brightness


def getContoours(img, imgContour):
    biggest = np.array([])
    maxArea = 0
    contours, hierarcy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area >50:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0),3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 3)
    x, y, w, h = cv2.boundingRect(biggest)
    #print([x,y,w,h])
    cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 155, 0), 2)
    cv2.putText(imgContour, f"{x},{y}",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2)
    cv2.putText(imgContour, f"{x+w},{y}",
                (x+w, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2)
    cv2.putText(imgContour, f"{x},{y+h}",
                (x, y+h), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2)
    cv2.putText(imgContour, f"{x+w},{y+h}",
                (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2)

    return biggest

def reorder(mypoints):
    mypoints = mypoints.reshape((4,2))
    mypointsNew = np.zeros((4,1,2), np.int32)
    add = mypoints.sum(1)
    #print("add",add)
    mypointsNew[1] = mypoints[np.argmin(add)]
    mypointsNew[2] = mypoints[np.argmax(add)]
    diff = np.diff(mypoints,axis=1)
    #print("diff",diff)
    mypointsNew[3] = mypoints[np.argmin(diff)]
    mypointsNew[0] = mypoints[np.argmax(diff)]
    #print("NewPoints ", mypointsNew)
    return mypointsNew


def getWarp(img,biggest):
    biggest = reorder(biggest)
    print(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img,matrix,(widthImg,heightImg))

    imgCropped =  imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(widthImg,heightImg))
    return imgCropped


def preprocessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernal = np.ones((5,5))
    imgDilate = cv2.dilate(imgCanny,kernal,iterations=2)
    imgErode = cv2.erode(imgDilate,kernal,iterations=1)

    return imgErode

while True:
    #success, img = cap.read()
    img = cv2.imread("resources/paper.jpg")
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()
    imgTres = preprocessing(img)
    biggest = getContoours(imgTres,imgContour)
    #print(biggest)

    imgWrapped = getWarp(img,biggest)
    cv2.imshow("Stacked",imgWrapped)
    cv2.imshow("Contour",imgContour)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
