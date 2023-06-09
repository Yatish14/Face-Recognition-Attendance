#DIP Project

#Team Member 1 : P.Yatish Chandra
#Roll Number 1 : CS20B1045

#Team Member 2 : Sai Kiran Reddy
#Roll Number 2 : CS20B1059

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#Images folder contains the images of the persons faces
path = 'images'

#Storing all the images in an array and giving corresponding class names as the Image Name
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

#Function to find the encoding of face which is 128 dimensional feature vector using inbuilt function face_encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#Function to mark the Attendance of respective person in a CSV File
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

#Getting the face encodings of all the images in images folder
encodeListKnown = findEncodings(images)
print('Encoding Complete')

#Opening the webcam of device
cap = cv2.VideoCapture(1)

while True:
    #Reading a frame from the video
    success, img = cap.read()
    #Resizing the frame for speeding up the face recognition
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    #Conversion from BGR to RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    #Getting the location of face and its encodings in the current video frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodingsCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    #for every faces in the current frame check if there is any match with the known faces
    for encodeFace, faceLoc in zip(encodingsCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            #Getting name of the person from class name 
            name = classNames[matchIndex].upper()

            #Drawing a box around the faces in current frame and writing name on it
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()