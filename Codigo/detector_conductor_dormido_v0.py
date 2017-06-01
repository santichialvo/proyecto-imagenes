# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:38:37 2017

@author: etekken
"""

# Version 1.0 del código de detección de conductor dormido.
# Para más detalles sobre la detección del ojo cerrado ver Paper_Blink.pdf
# python detector_conductor_dormido.py --predictor shape_predictor_68_face_landmarks.dat --sonido alarma.wav

import pyglet
import utils
import cv2
import time
import dlib
import argparse
from WebcamThread import WebcamThread
 
# threshold para indicar pestañeo (o ojo cerrado)
const_thres = 0.3
# ṕor cuantos frames aceptamos que este cerrado
const_cons_frames = 40

# contador de frames
var_counter = 0
# booleano para indicar si la alarma esta activada
var_alarma = False
# variable para debug
debug = False


# argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--predictor", required=True)
ap.add_argument("-s", "--sonido")
args = vars(ap.parse_args())

musica = pyglet.media.load(args["sonido"], streaming=False)

# agarro los puntos correspondientes al ojo izquierdo y derecho
(izqIni, izqFin) = (42,48)
(derIni, derFin) = (36,42)

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

    # loop sobre los rectangulos detectados (por si hay mas de una cara)
    for rect in rects:
        
        # detecto los rasgos faciales
        shape = predictor(gray, rect)
        # convierto las coordenadas (x, y) a tipo numpy (para poder dibujar)
        shape = utils.convertir_shape(shape)
        
        # extraigo las coordenadas que corresponden al ojo izq y der
        ojoIzq = shape[izqIni:izqFin]
        ojoDer = shape[derIni:derFin]
        # calculo el rao
        izqRAO = utils.radio_aspecto_ojo(ojoIzq)
        derRAO = utils.radio_aspecto_ojo(ojoDer)
        
        # promedio, lo dice el paper
        rao = (izqRAO+derRAO)/2.0
        
        if debug:
            print rao
        
        # convex hull para visualizar
        izqHull = cv2.convexHull(ojoIzq)
        derHull = cv2.convexHull(ojoDer)
        cv2.drawContours(frame, [izqHull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [derHull], -1, (0, 255, 255), 1)

        # comparo el rao con el threshold
        if rao <= const_thres:
            var_counter += 1
            
            if var_counter >= const_cons_frames:
                # para que no suene si ya esta sonando
                if not var_alarma:
                    var_alarma = True
                  
                    if args["sonido"] != "":
                        musica.play()
                
                cv2.putText(frame, "TE DORMISTE!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # sino, resetear
        else:
            var_counter = 0
            var_alarma = False
        
    # mostrar el frame    
    cv2.imshow("Frame", frame)
    
    # detecto si se pulso alguna tecla
    key = cv2.waitKey(1)
    
    # si apreto la q, salir del loop
    if key == ord("q"):
        break
    
# limpiar
vs.stop()
cv2.destroyAllWindows()