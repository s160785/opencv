#Importing Modules
import numpy as np
import cv2
import face_recognition as fr
import glob
import pathlib

face_locations = {}
face_encodings = {}

training_set_list = list(glob.glob("image_Classification/training_images/*/*"))
training_set_list = [i.replace('\\','/') for i in training_set_list]

lis = [cv2.imread(i).shape for i in training_set_list]
print(np.argmin(lis))
print(lis)
size = min(lis)
print(size)

print(training_set_list[0])
for image in training_set_list:
    label = image.split('/')[2]
    img = fr.load_image_file(str(image))
    img = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)),interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    face_locations[label] = fr.face_locations(img)[0]
    face_encodings[label] = fr.face_encodings(img)[0]
    # cv2.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

    # cv2.imshow("img",img)
    # cv2.waitKey(0)
labels = list(face_encodings.keys())
img_Test = fr.load_image_file("resources\colt2.jpg")
img_Test = cv2.resize(img_Test,(int(img_Test.shape[1]/2),int(img_Test.shape[0]/2)),interpolation=cv2.INTER_AREA)
img_Test = cv2.cvtColor(img_Test,cv2.COLOR_BGR2RGB)
faceLocTest = fr.face_locations(img_Test)[0]
encodeimg_Test = fr.face_encodings(img_Test)[0]
print(faceLocTest)
cv2.rectangle(img_Test,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

known_faces = list(face_encodings.values())

results = fr.compare_faces(known_faces,encodeimg_Test,tolerance =0.5)
faceDis = fr.face_distance(known_faces,encodeimg_Test)
print(results,faceDis)
name = labels[np.argmin(faceDis)]
if results[np.argmin(faceDis)]:
    cv2.putText(img_Test,f'{name} {round(min(faceDis),2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
    img =cv2.imread(training_set_list[np.argmin(faceDis)])
    cv2.imshow("Match Image",img)
    cv2.imshow('Result',img_Test)
    cv2.waitKey(0)
else:
    cv2.putText(img_Test,'No Match',(50,50),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    cv2.imshow('Result',img_Test)
    cv2.waitKey(0)
cv2.destroyAllWindows()
