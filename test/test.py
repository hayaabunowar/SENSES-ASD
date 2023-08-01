import os
import cv2
import pytest 
import imutils
import unittest
import requests
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from werkzeug.datastructures import FileStorage
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import FERWebApp


FERWebApp.config['UPLOAD_FOLDER'] = 'uploads'

emotion_list = ["angry" ,"disgust","fear", "happy", "sad", "surprised","neutral"]

face_detection_path = os.path.abspath("haarcascade_frontalface_default.xml")
face_detection_model = cv2.CascadeClassifier(face_detection_path)

emotion_model_path = os.path.abspath("_mini_XCEPTION.102-0.66.hdf5")
emotion_detection_model = load_model(emotion_model_path, compile=False)

# def test_face_detection():
#     image = cv2.imread('face-detection--negative-img.jpg')
#     gray_face = face_detection_model.detectMultiScale(image,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
#     assert len(gray_face) > 0, "No face detected in the test image"



# def test_emotion_detection():
#     image = cv2.imread("testImages/emotion-detection-img-neutral.jpg")
#     image_frame = imutils.resize(image,width=300)
#     gray_img = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
#     gray_face = face_detection_model.detectMultiScale(image,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
#     if len(gray_face) > 0:
#         gray_face = sorted(gray_face, reverse=True,
#         key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
#         (fX, fY, fW, fH) = gray_face
#         roi = gray_img[fY:fY + fH, fX:fX + fW]
#         roi = cv2.resize(roi, (64, 64))
#         roi = roi.astype("float") / 255.0
#         roi = img_to_array(roi)
#         roi = np.expand_dims(roi, axis=0)
#         roi= np.expand_dims(roi, axis=-1)
#     preds = emotion_detection_model.predict(roi)[0]
#     label = emotion_list[preds.argmax()]

#     # Check if the predicted emotion is correct
#     assert label == "neutral", "The emotion detection model did not correctly identify the emotion"


# def test_app_instance() :
#     assert FERWebApp is not None


# def test_routes():
#     client = FERWebApp.test_client()
#     resp = client.get('/')
#     assert resp.status_code == 200
#     resp = client.get('/realtimeVideo')
#     assert resp.status_code == 200
#     resp = client.get('/videoUpload')
#     assert resp.status_code == 200

# @pytest.fixture
# def client():
#     app = FERWebApp
#     app.config['TESTING'] = True
#     with app.test_client() as client:
#         yield client

# def test_file_handling_functionality(client):
#     with open('testImages/emotion-detection-img-happy.jpg', 'rb') as f:
#         data = {'imgFileSubmission': (f, 'emotion-detection-img-happy.jpg')}
#         response = client.post('/', data=data, content_type='multipart/form-data')
#     assert response.status_code == 200
    
#     upload_dir = os.path.join(FERWebApp.root_path, 'uploads')
#     assert os.path.exists(os.path.join(upload_dir, 'emotion-detection-img-happy.jpg'))


def test_video_upload(client):
    # open test video file
        with open('testImages/vidUpload-FER-happy.mp4', 'rb') as f:
            data = {'videoFileSubmission': (f, 'vidUpload-FER-happy.mp4')}
            response = client.post('/videoUpload', data=data, content_type='multipart/form-data')
        #read video frames
        capture = cv2.VideoCapture(os.path.join(FERWebApp.root_path, 'uploads', 'vidUpload-FER-happy.mp4'))
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_processed = 0
        label=None
        while frames_processed < num_frames:
            video_frame = capture.read()[1]
            if video_frame is not None: 
                video_frame = imutils.resize(video_frame,width=300)
                gray_img = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
                gray_face = face_detection_model.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
                if len(gray_face) > 0:
                    gray_face = sorted(gray_face, reverse=True,
                    key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                    (fX, fY, fW, fH) = gray_face
                    roi = gray_img[fY:fY + fH, fX:fX + fW]
                    roi = cv2.resize(roi, (64, 64))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    predictions = emotion_detection_model.predict(roi)[0]
                    label = emotion_list[predictions.argmax()]

        # Check if the predicted emotion is correct
                    assert label == "happy", "The emotion detection model did not correctly identify the emotion"
            frames_processed += 1
                    