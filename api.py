import os
import sys
import time
import dlib
import cv2
import imutils
import uvicorn
import logging
import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel
from imutils import face_utils
from scipy.spatial import distance as dist
from fastapi import FastAPI, File, UploadFile

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
COUNTER = 0
msg = "None"

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

class vector(BaseModel):
    status : str

app = FastAPI()


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

def detect(frame):
	frame1 = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
	minNeighbors=5, minSize=(30, 30),
	flags=cv2.CASCADE_SCALE_IMAGE)

	for (x, y, w, h) in rects:

		rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		eye = final_ear(shape)
		ear = eye[0]
		leftEye = eye [1]
		rightEye = eye[2]
		distance = lip_distance(shape)
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		lip = shape[48:60]
		cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

		if ear < EYE_AR_THRESH:
			return "drowsy"

		elif(distance > YAWN_THRESH):
			return "yawn"
		else:
			return "active"







@app.post('/post_user',response_model=vector)
async def post_user(image :UploadFile = File(...)):
    try:
        if '.npy' in image.filename:
            image = np.load(img.file)
        else:
            contents = image.file.read()
            nparr = np.fromstring(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        msg = detect(image)
        return {"status": msg}
    except:
    	return {"status":"Internal server error"}


if __name__=="__main__":
    uvicorn.run("api:app",port=52112,log_level="info")  

