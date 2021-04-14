import numpy as np
import cv2
import face_recognition as fr

img = fr.load_image_file("resources\el5n9i.jpg")
img = cv2.resize(img,(320,640))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
faceLoc = fr.face_locations(img)[0]
encodeimg = fr.face_encodings(img)[0]
print(faceLoc)
cv2.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

img2 = fr.load_image_file("resources\Malvika.jpg")
img2 = cv2.resize(img2,(640,320))
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
faceLoc2 = fr.face_locations(img2)[0]
encodeimg2 = fr.face_encodings(img2)[0]
print(faceLoc2)
cv2.rectangle(img2,(faceLoc2[3],faceLoc2[0]),(faceLoc2[1],faceLoc2[2]),(255,0,255),2)

img3 = fr.load_image_file("resources\malavika-sharma-4.jpg")
img3 = cv2.resize(img3,(320,640))
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
faceLoc3 = fr.face_locations(img3)[0]
encodeimg3 = fr.face_encodings(img3)[0]
print(faceLoc3)
cv2.rectangle(img3,(faceLoc3[3],faceLoc3[0]),(faceLoc3[1],faceLoc3[2]),(255,0,255),2)

img_Test = fr.load_image_file("resources\Malvika-Sharma-2019.jpg")
img_Test = cv2.resize(img_Test,(640,320))
img_Test = cv2.cvtColor(img_Test,cv2.COLOR_BGR2RGB)
faceLocTest = fr.face_locations(img_Test)[0]
encodeimg_Test = fr.face_encodings(img_Test)[0]
print(faceLocTest)
cv2.rectangle(img_Test,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = fr.compare_faces([encodeimg,encodeimg2,encodeimg3],encodeimg_Test)
faceDis = fr.face_distance([encodeimg,encodeimg2,encodeimg3],encodeimg_Test)
print(results,faceDis)
cv2.putText(img_Test,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Image3',img3)
cv2.imshow('Image2',img2)
cv2.imshow('Image',img)
cv2.imshow('Test Image',img_Test)
cv2.waitKey(0)