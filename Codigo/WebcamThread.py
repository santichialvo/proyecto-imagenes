# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:08:59 2017

@author: etekken
"""

from threading import Thread
import cv2

class WebcamThread:
	def __init__(self):
		# iniciar la camara por default y leer el primer frame
		self.video = cv2.VideoCapture(0)
		(self.grabbed, self.frame) = self.video.read()

		# para parar el thread
		self.stopped = False

	def start(self):
		# comienzo el thread y le paso como target la funcion update
		t = Thread(target=self.update)
		t.daemon = True
		t.start()
		return self

	def update(self):
		# loop infinito hasta que la bandera cambie
		while True:
			if self.stopped:
				return
			(self.grabbed, self.frame) = self.video.read()

	def read(self):
		return self.frame

	def stop(self):
		self.stopped = True