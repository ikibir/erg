import os, psutil
pid = os.getpid()
py = psutil.Process(pid)

import cv2
from model import FacialExpressionModel
import numpy as np
from threading import Thread
import time
queue = []

rgb = cv2.VideoCapture("videoplayback.mp4")
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
program_ended = False
pred_thread_started, read_thread_started = False, False

def get_frames():
    global program_ended, pred_thread_started, read_thread_started
    frame_count = 0
    while not program_ended:
        read_thread_started = True
        if pred_thread_started:
            ret, fr = rgb.read()
            if ret:
                frame_count += 1
                queue.append(fr)
                time.sleep(0.029)
            else:
                program_ended = True

def prediction():
    global program_ended, pred_thread_started, read_thread_started
    model = FacialExpressionModel("face_model.json", "face_model.h5")

    def detect_faces(fr):
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray, 1.3, 5)
        return gray, faces

    def draw_result(image, faces, preds):
        for (pred, (x, y, w, h)) in zip(preds, faces):
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(image, pred, (x, y), font, 1, (255, 255, 0), 2)

    def predict(gray, faces):
        preds = []
        for (x, y, w, h) in faces:
            fc = gray[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            preds.append(pred)
        
        return preds

    while not program_ended:
        pred_thread_started = True
        if read_thread_started:
            if len(queue) > 0:
                frame = queue.pop(0)
                gray, faces = detect_faces(frame)
                pred = predict(gray, faces)
                draw_result(frame, faces, pred)
               
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) == 27:
                    program_ended = True
                    break


camera_thread = Thread(target=get_frames)
predic_thread = Thread(target=prediction)

camera_thread.daemon = True
predic_thread.daemon = True
camera_thread.start()
predic_thread.start() #prediction()

while not program_ended:
    if pred_thread_started:
        print("Queue Size: ", len(queue))
        print(f"Allocated Size: {(py.memory_info()[0]/2.**30)*1024} mb")
    time.sleep(0.1)
program_ended = True