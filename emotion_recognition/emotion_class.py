import numpy as np
import cv2
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
import matplotlib.pyplot as plt


class emotionRecognizer:
    def __init__(self):
        self.img_rows = 48
        self.img_cols = 48

        self.face_cascade = cv2.CascadeClassifier('saved_model/haarcascade_frontalface_default.xml')
        self.modelFile = "saved_model/opencv_face_detector_uint8.pb"
        self.configFile = "saved_model/opencv_face_detector.pbtxt"

        self.emotion = ['Angry', "Disgust", 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.model = []

    def loadModel(self):
        for i in range(2):
            m = load_model('saved_model/' + 'cnn'+str(i)+'.h5')
            print(str(i+1) + '/3')
            self.model.append(m)
        
        m = load_model('saved_model/ensemble.h5')
        self.model.append(m)

        print('3/3')
        print('Loading Complete!')
    
    def predict(self, x):
        x_rev = np.flip(x, 1)
        x = x.astype('float32')
        x_rev = x_rev.astype('float32')
        x /= 255
        x_rev /= 255
        p = np.zeros((1, 14))
        p[:,0:7] = self.model[0].predict(x.reshape(1,48,48,1))
        p[:,7:14] = self.model[1].predict(x_rev.reshape(1,48,48,1))
        pre = self.model[2].predict(p)
        return pre

    def getFaces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = []

        faces_detected = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for i, (x,y,w,h) in enumerate(faces_detected):
            my = int(y + h/2)
            mx = int(x + w/2)

            if h<w:
                c = int(h/2)
            else:
                c = int(w/2)

            face = gray[my-c : my+c, mx-c:mx+c]
            try:
                face_48 = cv2.resize(face,(48,48), interpolation = cv2.INTER_CUBIC)
                faces.append((y, x+w, face_48))
            
            except:
                pass

        return faces

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap.open()

emotion_recog = emotionRecognizer()

while(True):
    ret, img = cap.read()
    
    faces = emotion_recog.getFaces(img)

    for i, (y,x,face) in enumerate(faces):
        pre = emotion_recog.predict(face)
        k = np.argmax(pre)
        txt = emotion[k] + ' [' + str(int(pre[0,k]*100)) + '%]'
        cv2.putText(img, txt, (x, y), font, 1.0,(0,255,0),2,cv2.LINE_AA)
        # cv2.imshow(str(i), face)

    cv2.imshow('Camera', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

        
    