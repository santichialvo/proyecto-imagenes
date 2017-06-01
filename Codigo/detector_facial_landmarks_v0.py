# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:04:59 2017

@author: etekken
"""

#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

import utils
import cv2
import time
import dlib
import argparse
from WebcamThread import WebcamThread

# argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--predictor", required=True)
args = vars(ap.parse_args())

# iniciar el detector de caras de dlib
print("Inicializando detector de caras...")
detector = dlib.get_frontal_face_detector()
print("Cargando detector de rasgos faciales...")
predictor = dlib.shape_predictor(args["predictor"])
    
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
        
        # detecto los rasgos faciales
        shape = predictor(frame, rect)
        
        # convierto las coordenadas (x, y) a tipo numpy (para poder dibujar)
        shape = utils.convertir_shape(shape)
        
        # loop y dibujo
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        
    # mostrar el frame    
    cv2.imshow("Frame", frame)
    
    # detecto si pulso alguna tecla
    key = cv2.waitKey(1)
    
    # si apreto la q, salir del loop
    if key == ord("q"):
        break