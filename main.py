# importing libraries
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2 
import numpy as np

# importing from songs_recommender.py
import songs_recommender as sr

# declaring classifier to detect face
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# declaring classifier to classify emotions based on face detected
emotion_classifier = load_model('emotion_classifier.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotions_detected = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}
final_emotion_detected = None

# starting webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = emotion_classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            emotions_detected[label] += 1
            final_emotion_detected = label
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow('Emotion Detector',frame)

    # close webcam if 'x' is clicked
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()


# Neutral is neglected since most of the time the emotion is Neutral
del emotions_detected['Neutral']
# most_frequent_emotion = max(emotions_detected, key=emotions_detected.get)
songs_recommended = sr.Songs_Recommendation(final_emotion_detected).get_songs_recommended()
print(songs_recommended)