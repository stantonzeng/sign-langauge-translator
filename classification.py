import picTakingScript

import numpy as np

import matplotlib.pyplot as plt
import os

import cv2
import time
import uuid

from sklearn.metrics import confusion_matrix , classification_report

import numpy as np

import tensorflow as tf
from tensorflow import keras

CATEGORIES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
              "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

def getPrediction(snapshot, cnn):
    snapshot = np.array(snapshot).reshape(-1, 80, 80, 1)
    snapshot = snapshot/255

    #print(snapshot.shape)

    sign_prediction = cnn.predict(snapshot)

    return CATEGORIES[np.argmax(sign_prediction)]

def main():

    cnn = tf.keras.models.load_model("models/cnn_v11") #change this according to the saved model

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    board = np.zeros((480, 800, 3), dtype=np.uint8)

    previous_guess = " "
    word = ""
    pTime = time.time()
    while True:
        cTime = time.time()
        
        success, img = cap.read()

        cv2.imshow("Translator", img)
        
        img_array = cap.read()[1]
        grayImage = cv2.cvtColor(img_array, cv2.IMREAD_GRAYSCALE)
        snapshot = cv2.resize(grayImage[:, :, :1], (80, 80))
        guess = getPrediction(snapshot, cnn)

        if previous_guess != guess:
            cv2.rectangle(board, (180, 15), (220,50), (0,0,0), thickness = cv2.FILLED)
            previous_guess = guess
            pTime = cTime
        
        cv2.imshow("Board", board)

        if cTime > pTime + 3:
            word = word + guess
            pTime = cTime
        
        cv2.putText(board, "Prediction: {}".format(guess), (0,40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))
        cv2.putText(board, "{}".format(word), (0,100), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()