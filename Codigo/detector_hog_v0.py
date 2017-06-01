# -*- coding: utf-8 -*-
"""
Created on Tue May 30 20:59:53 2017

@author: etekken
"""

#   This face detector is made using the now classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image
#   pyramid, and sliding window detection scheme. This type of object detector
#   is fairly general and capable of detecting many types of semi-rigid objects
#   in addition to human faces.
#   Ver paper_HOG

import utils
import cv2
import time
import dlib
from WebcamThread import WebcamThread

# iniciar el detector de caras de dlib
print("Inicializando detector de caras...")
detector = dlib.get_frontal_face_detector()
    
# comenzar el thread de video
print("Comenzando el thread de video...")
vs = WebcamThread()
vs.start()
time.sleep(1.0)

# loop por todos los frames
while True:
	# leer el frame, hacerle un resize y convertirla a gris
    frame = vs.read()
    frame = utils.resize(frame, wr=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectar caras en el frame gris
    rects = detector(gray, 0)

    # loop sobre los rectangulos detectados
    for rect in rects:
        # dibujar el rectangulo
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 255), 1)
        
    # mostrar el frame    
    cv2.imshow("Frame", frame)
    
    # detecto si pulso alguna tecla
    key = cv2.waitKey(1)   
    
    # si apreto la q, salir del loop
    if key == ord("q"):
        break