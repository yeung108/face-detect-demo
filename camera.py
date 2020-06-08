import sys
import functools
import numpy as np
import cv2
import base64
import time
from queue import Queue
import pandas as pd

from gaze_tracking import GazeTracking
from datetime import datetime
import dlib
import os
import platform

BG_COLOR = 0
MIN_FACE_SIZE = 60
encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
detect_encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
model_directory = 'data'

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
age_list = [1, 5, 10, 17.5, 28.5, 40.5, 50.5, 80]
gender_list = ['Male', 'Female']
# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class NormalCamera():
    instance = None
    def __init__(self, testMode, rotation):
        self.blinked = False
        self.captured = False
        self.rotation = rotation
        self.photoList = dict()
        self.genderList = list()
        self.ageList = list()
        self.timeList = list()

        # initialize the frame dimensions (we'll set them as soon as we read
        # the first frame from the video)
        self.W = None
        self.H = None

        # Model
        self.face_cascade = cv2.CascadeClassifier()
        if not self.face_cascade.load('haarcascade_frontalface_alt.xml'):
            print("face detector model not loaded")
        self.age_net = cv2.dnn.readNetFromCaffe(model_directory+'/deploy_age.prototxt', model_directory+'/age_net.caffemodel')
        self.gender_net = cv2.dnn.readNetFromCaffe(model_directory+'/deploy_gender.prototxt', model_directory+'/gender_net.caffemodel')

        # Edit this for development
        print("Running on "+platform.system())
        if platform.system() == 'Linux' or platform.system() == "linux2":
            self.cam = cv2.VideoCapture(-1)
        else:
            self.cam = cv2.VideoCapture(1)
        self.cam.set(3,1920)
        self.cam.set(4,1920)
        self.gaze = GazeTracking(testMode)

    def __initVar__(self):
        self.blinked = False
        self.captured = False
    
    def __startDetect__(self, liveliness, ageWeightedAverage, detectTimeout=50.0, savePhoto=False):
        detected = False
        start_time = time.process_time()
        if (savePhoto):
            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d%m%Y_%H%M%S")
            # Create target Directory if don't exist
            if not os.path.exists(dt_string):
                os.mkdir(dt_string)
                print("Directory " , dt_string ,  " Created ")
            else:    
                print("Directory " , dt_string ,  " already exists")
        try:
            while not detected:
                timedout = (time.process_time() - start_time) >= detectTimeout
                currentTime = str(format(time.process_time(), '.4f'))
                print(time.process_time() - start_time)
                # Check timeout
                if (not timedout):
                    aligned_frames = next(self.get_aligned_frames())

                    gray_image = cv2.cvtColor(aligned_frames, cv2.COLOR_BGR2GRAY)
                    largest_face = self.face_detect(gray_image)
                    x, y, w, h = largest_face

                    if (x >= 0 and y >= 0 and w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE):

                        # Check Blinking
                        if (self.gaze.is_blinking()):
                            self.blinked = True
                            
                        # Capture if 1) Not liveliness or 2) livenlincess and blinked before and not blinking
                        if ((self.blinked and not(self.gaze.is_blinking())) or not liveliness):
                            # take picture when not blinking
                            # Get Face 
                            face_img = aligned_frames[y:y+h, x:x+w].copy()
                            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                            # Predict Gender
                            self.gender_net.setInput(blob)
                            gender_preds = self.gender_net.forward()
                            gender = gender_list[gender_preds[0].argmax()]
                            # self.genderList[gender] += 1
                            self.genderList.append(gender)

                            # Predict Age
                            self.age_net.setInput(blob)
                            age_preds = self.age_net.forward()
                            pred = 0.00
                            sumOfValue = float(age_preds[0].sum())
                            for index, value in enumerate(age_list):
                                pred += float(value) * float(age_preds[0][index]) / sumOfValue
                            if (ageWeightedAverage):
                                age = pred
                            else: 
                                age = age_list[age_preds[0].argmax()]
                            self.ageList.append(age)

                            # time
                            self.timeList.append(currentTime)

                            ret, jpeg = cv2.imencode('.jpg', aligned_frames, encode_params)
                            self.photoList[currentTime] = jpeg.tobytes()

                            if (savePhoto):
                                cv2.rectangle(aligned_frames, (x, y), (x+w, y+h), (255, 0, 0), 2)
                                cv2.imwrite(os.path.join(dt_string, currentTime+'_'+str(age)+'_'+str(gender)+'.jpg'), aligned_frames, encode_params)

                            self.__initVar__()
                    else:
                        continue
                else:
                    raise Exception("Detection Timeout")
        except Exception as e:
            print(e, file=sys.stderr)
    
    def __endDetect__(self):
        data = {'Gender': self.genderList, 'Age': self.ageList, 'Time': self.timeList}
        print("data")
        print(data)
        df = pd.DataFrame(data)
        print("df")
        print(df)
        if (not df.empty):
            print("test")
            genderMax = df['Gender'].mode()[0]
            ageMean = df[df.Gender == genderMax]['Age'].mean()
            print(ageMean)
            print("mean")
            print(ageMean)
            print("gender max")
            print(df['Gender'].mode()[0])
            # print(df.mode(axis='columns')['Gender'].argmax())
            print("closest value")
            closestTime = df.iloc[(df['Age']-ageMean).abs().argsort()[0]]['Time']
            print(closestTime)
            print("df.describe()")
            print(df.describe())
            print("self.photoList.keys")
            print(self.photoList.keys())
            # Get closest value of photo
            photo = self.photoList.get(closestTime)
            self.photoList = dict()
            self.genderList = list()
            self.ageList = list()
            self.timeList = list()
            return base64.b64encode(photo).decode(), genderMax, ageMean
        else:
            return None, None, None

    def __del__(self):
        self.release()

    def release(self):
        self.cam.release()

    @classmethod
    def get_instance(cls, testMode, rotation):
        if cls.instance is None:
            cls.instance = NormalCamera(testMode, rotation)
        return cls.instance

    def get_aligned_frames(self):
        while True:
            ret, frame = self.cam.read()
            if (self.rotation in [0,1,2]):
                frame = cv2.rotate(frame, self.rotation)
            # if the frame dimensions are empty, set them
            if self.W is None or self.H is None:
                (self.H, self.W) = frame.shape[:2]
            self.gaze.refresh(frame)
            yield frame

    def get_jpeg_frame(self):
        for aligned_frames in self.get_aligned_frames():
            ret, jpeg = cv2.imencode('.jpg', aligned_frames, encode_params)
            return jpeg.tobytes()

    def get_jpeg_depth_frame(self):
        for aligned_frames in self.get_aligned_frames():
            ret, jpeg = cv2.imencode('.jpg', aligned_frames, encode_params)
            return jpeg.tobytes()

    def face_detect(self, gray):
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 2)
        return functools.reduce(lambda x, y: x if x[2]*x[3] > y[2]*y[3] else y, faces, (-1,-1, 0, 0))

    def video_feed(self, liveliness, ageWeightedAverage):
        for aligned_frames in self.get_aligned_frames():
            gray_image = cv2.cvtColor(aligned_frames, cv2.COLOR_BGR2GRAY)
            largest_face = self.face_detect(gray_image)
            x, y, w, h = largest_face

            cv2.rectangle(aligned_frames, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if self.gaze.is_blinking():
                cv2.putText(aligned_frames, "BLINKING", (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
            else: 
                cv2.putText(aligned_frames, "NOT BLINKING", (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)

            if (x >= 0 and y >= 0 and w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE):

                # Check Blinking
                if (self.gaze.is_blinking()):
                    self.blinked = True
                    
                if (self.blinked and not(self.gaze.is_blinking()) or not liveliness):
                    # take picture when not blinking
                    self.captured = True
                    # Get Face 
                    face_img = aligned_frames[y:y+h, x:x+w].copy()
                    blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                    # Predict Gender
                    self.gender_net.setInput(blob)
                    gender_preds = self.gender_net.forward()
                    gender = gender_list[gender_preds[0].argmax()]

                    # Predict Age
                    self.age_net.setInput(blob)
                    age_preds = self.age_net.forward()
                    pred = 0.00
                    sumOfValue = float(age_preds[0].sum())
                    for index, value in enumerate(age_list):
                        pred += float(value) * float(age_preds[0][index]) / sumOfValue
                    if (ageWeightedAverage):
                        age = pred
                    else: 
                        age = age_list[age_preds[0].argmax()]
                    cv2.putText(aligned_frames, str(age)+' '+gender ,(x,y), cv2.FONT_HERSHEY_DUPLEX, 1,(255,0,0),2,cv2.LINE_AA)
                    self.__initVar__()

            ret, jpeg = cv2.imencode('.jpg', aligned_frames, encode_params)
            return jpeg.tobytes()

    def detect(self, full_image=False, jpeg_bytes=False, detectTimeout=50.0):
        detected = False
        start_time = time.process_time()
        try:
            while not detected:
                timedout = (time.process_time() - start_time) >= detectTimeout
                # Check timeout
                if (not timedout):
                    # aligned_frames = next(self.get_aligned_frames())

                    # detected_image = self.get_jpeg_frame()
                    aligned_frames, detected_image = self.get_motion_detection_frame()
                    return detected_image
                    # gray_image = cv2.cvtColor(aligned_frames, cv2.COLOR_BGR2GRAY)
                    # largest_face = self.face_detect(gray_image)
                    # x, y, w, h = largest_face

                    # # Check Face
                    # if (x >= 0 and y >= 0 and w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE):

                    #     # Check Blinking
                    #     if (self.gaze.is_blinking()):
                    #         self.blinked = True

                    #     if (self.blinked and not(self.gaze.is_blinking())):
                    #         # take picture when not blinking
                    #         detected = True

                    #         self.blinked = False
                    #         return detected_image
                    #     else:
                    #         continue
                    # else:
                    #     continue
                else:
                    raise Exception("Detection Timeout")
        except Exception as e:
            print(e, file=sys.stderr)
        return None
