import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from time import sleep, strftime
import requests
import pandas as pd

face_classifier = cv2.CascadeClassifier('Emotion-Detection-master\haarcascade_frontalface_default.xml')
classifier = load_model('D:\Emotion-Detection-master\Emotion-Detection-master\Emotion_Detection.h5')

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

# Initialize an empty DataFrame to store the data
data = pd.DataFrame(columns=['Time', 'Emotion'])

while True:
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # Save the detected emotion data
            current_time = strftime("%Y-%m-%d %H:%M:%S")
            data = data.append({'Time': current_time, 'Emotion': label}, ignore_index=True)

            # Save the emotion picture (optional)
            # cv2.imwrite(f'{label}_{current_time}.jpg', frame)

            # Send the detected emotion to the Flask server (optional)
            # emotion_data = {"emotion": label}
            # server_url = "http://your_server_ip:5000/emotion_data"
            # try:
            #     response = requests.post(server_url, json=emotion_data)
            #     print("Server response:", response.json())
            # except requests.exceptions.RequestException as e:
            #     print("Error sending data to the server:", e)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save data to an Excel sheet
data.to_excel('emotion_data.xlsx', index=False)

cap.release()
cv2.destroyAllWindows()
