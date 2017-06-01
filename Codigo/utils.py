# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:49:20 2017

@author: etekken
"""

import cv2
import numpy as np
from scipy.spatial import distance

# resize usando un metodo de interpolacion
def resize(img, wr=None, hr=None):

    (ho, wo) = img.shape[:2]

    # factor de ancho 
    r = wr / float(wo)
    dim = (wr, int(ho * r))

    # resize usando inter_area, usa resampling de area de pixel
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized

# convertir los 68x2 puntos de rasgos a un array numpy 
def convertir_shape(shape):
    # creo un array de ceros de (68,2)
    coords = np.zeros((68, 2), dtype="int")
    
    # para c/u de los 68 rasgos, detecto las coordenadas (x,y)
    # y las guardo en un array
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

# calcula RAO (Relacion de aspecto del ojo) de un ojo
def radio_aspecto_ojo(ojo):
    if ojo.size==0:
        return -1
    # distancias verticales
    A = distance.euclidean(ojo[1], ojo[5])
    B = distance.euclidean(ojo[2], ojo[4])

    # distancias horizontales
    C = distance.euclidean(ojo[0], ojo[3])

    # calcular el radio de aspecto del ojo y devolverlo
    rao = (A+B)/(2.0*C)

    return rao