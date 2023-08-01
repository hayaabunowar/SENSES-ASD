import os
from pathlib import Path
import sys
from gtts import gTTS
from pygame import mixer
from pygame import time
import mmap
import imutils
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
os.environ['KERAS_BACKEND']='tensorflow'


app_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the model
face_detection_path = os.path.abspath(os.path.join(app_dir, "faceDetection", "haarcascade_frontalface_default.xml"))
emotion_model_path = os.path.join(app_dir, "modelTraining", "_mini_XCEPTION.102-0.66.hdf5")



#ASSIGNING MODELS TO VARIABLES WITHOUT COMPILATION
face_detection_model = cv2.CascadeClassifier(face_detection_path)
emotion_detection_model = load_model(emotion_model_path, compile=False)
emotion_list = ["angry" ,"disgust","fear", "happy", "sad", "surprised","neutral"]


#CREATING FLASK WEB APP INSTANCE 
FERWebApp = Flask(__name__)

#CONFIGURING UPLOADS FOLDER FOR WEB APP
FERWebApp.config['UPLOAD_FOLDER'] = 'uploads'



# REALTIME EMOTION RECOGNTION
@FERWebApp.route('/realtimeVideo', methods=['GET', 'POST'])
def realtime():
    if request.method == 'POST':
        cv2.namedWindow('Webcam')
        camera = cv2.VideoCapture(0)
        while True:
            video_frame = camera.read()[1]
            video_frame = imutils.resize(video_frame,width=300)
            gray_img = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
            gray_face = face_detection_model.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            canvas = np.zeros((250, 300, 3), dtype="uint8")
            frame_copy = video_frame.copy()
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
            else: continue
 
            for (i, (emotion, prob)) in enumerate(zip(emotion_list, predictions)):
                        text = "{}: {:.2f}%".format(emotion, prob * 100)
                        w = int(prob * 300)
                        cv2.rectangle(canvas, (7, (i * 35) + 5),(w, (i * 35) + 35), (255,255,0), -1)
                        cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255, 255, 255), 2)
                        cv2.putText(frame_copy, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,  (255,255,0), 2)
                        cv2.rectangle(frame_copy, (fX, fY), (fX + fW, fY + fH),
                                    (255,255,0), 2)
            cv2.imshow('Webcam', frame_copy)
            cv2.imshow("Probabilities", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

    return render_template('emotionWebApp.html')



# VIDEO EMOTION RECOGNITION
@FERWebApp.route('/videoUpload', methods=['GET', 'POST'])
def videoupload():
    if request.method == 'POST':

        #FILE HANDLING
        file = request.files['videoFileSubmission']
        filename=secure_filename(file.filename)
        upload_dir = os.path.join(FERWebApp.root_path, FERWebApp.config['UPLOAD_FOLDER'])
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)


        capture = cv2.VideoCapture(file_path)

        while True:
            video_frame = capture.read()[1]
            if video_frame is not None: 
                video_frame = imutils.resize(video_frame,width=300)
                gray_img = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
                gray_face = face_detection_model.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
                canvas = np.zeros((250, 300, 3), dtype="uint8")
                frame_copy = video_frame.copy()
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
                else: continue
                for (i, (emotion, prob)) in enumerate(zip(emotion_list, predictions)):
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),(w, (i * 35) + 35), (255,255,0), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255, 255, 255), 2)
                    cv2.putText(frame_copy, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,  (255,255,0), 2)
                    cv2.rectangle(frame_copy, (fX, fY), (fX + fW, fY + fH),(255,255,0), 2)
                cv2.imshow('Video', frame_copy)
                cv2.imshow("Probabilities", canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        capture.release()
        cv2.destroyAllWindows()
    
    return render_template('emotionWebApp.html')

    
# IMAGE EMOTION RECOGNITION
@FERWebApp.route('/', methods=['GET', 'POST'])
def main():

    if request.method == 'POST':

        #FILE HANDLING 
        file = request.files['imgFileSubmission']
        filename= secure_filename(file.filename)
        upload_dir = os.path.join(FERWebApp.root_path, FERWebApp.config['UPLOAD_FOLDER'])
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)

        image = cv2.imread(file_path)
        image_frame = imutils.resize(image,width=300)
        gray_img = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        gray_face = face_detection_model.detectMultiScale(image,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        if len(gray_face) > 0:
            gray_face = sorted(gray_face, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = gray_face
            roi = gray_img[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            roi= np.expand_dims(roi, axis=-1)
        preds = emotion_detection_model.predict(roi)[0]
        label = emotion_list[preds.argmax()]
        tts = gTTS(text='Expression is ... {}'.format(label), lang='en', slow=False)
        tts.save("Expression.mp3")
        with open('Expression.mp3') as file_save: 
            m = mmap.mmap(file_save.fileno(), 0, access=mmap.ACCESS_READ) 
        mixer.init()
        mixer.music.load(m)
        mixer.music.play()
        while mixer.music.get_busy():
            time.Clock().tick(30)
        else:
            m.close()
            os.remove('Expression.mp3')
    
    return render_template('emotionWebApp.html')    


if __name__ == "__main__":
    import webbrowser
    webbrowser.open('http://localhost:5000')
    FERWebApp.run()