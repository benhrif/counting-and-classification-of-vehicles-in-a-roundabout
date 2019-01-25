# import the necessary packages
from __future__ import print_function
from imutils.video import VideoStream
from PIL import Image as ImagePIL
from PIL import ImageTk
import tkinter as tki
from tkinter import filedialog
import threading
import datetime
import imutils
import cv2
import os
import time
import numpy as np
import sys
import os.path
import random
import csv
from pydarknet import Detector, Image

class PhotoBoothApp:
	def __init__(self,trackerType):
		# store the video stream object and output path, then initialize
		# the most recently read frame, thread for reading frames, and
		# the thread stop event
		self.videoPath=None
		self.isPlay=False
		self.isPause=False
		self.isRestart=False
		self.time = 0
		self.frameCounting = 0
		self.fps = 1

		
		# Initialize YOLO Detection parameters
		self.confThreshold = 0.1  #Confidence threshold
		self.nmsThreshold = 0.4   #Non-maximum suppression threshold
		self.inpWidth = 320       #Width of network's input image
		self.inpHeight = 320     #Height of network's input image
		
		self.sec = 0
		self.frameRate = 0.1

		self.net = Detector(bytes("yolov3.cfg", encoding="utf-8"), bytes("yolov3.weights", encoding="utf-8"), 0, bytes("coco.data",encoding="utf-8"))
		

		# initialize a dictionary that maps strings to their corresponding
		# OpenCV object tracker implementations
		self.OPENCV_OBJECT_TRACKERS = {
			"csrt": cv2.TrackerCSRT_create,
			"kcf": cv2.TrackerKCF_create,
			"boosting": cv2.TrackerBoosting_create,
			"mil": cv2.TrackerMIL_create,
			"tld": cv2.TrackerTLD_create,
			"medianflow": cv2.TrackerMedianFlow_create,
			"mosse": cv2.TrackerMOSSE_create
		}
		self.e1 = (0, 0, 0, 0)
		self.e2 = (0, 0, 0, 0)
		self.e3 = (0, 0, 0, 0)
		self.e4 = (0, 0, 0, 0)
		self.e5 = (0, 0, 0, 0)
		
		self.s1 = (0, 0, 0, 0)
		self.s2 = (0, 0, 0, 0)
		self.s3 = (0, 0, 0, 0)
		self.s4 = (0, 0, 0, 0)
		self.s5 = (0, 0, 0, 0)
		self.frame = None
		self.crop_frame = None
		self.thread = None
		self.stopEvent = None

		# initialize the root window and image panel
		self.root = tki.Tk()
		self.panel = None
		
		# initialize OpenCV's special multi-object tracker
		self.trackers = cv2.MultiTracker_create()         
		self.trackerType = trackerType 

		self.trackingMatrix = [[]]
		self.countTracker = 0

	
		btnDestroy = tki.Button(self.root, text="Cancel",command=self.root.destroy)
		btnDestroy.pack()
		btnDestroy.place(height=50, width=150, x=10, y=100)		
	
		btnRestart = tki.Button(self.root, text="Restart",command=self.restart)
		btnRestart.pack()
		btnRestart.place(height=50, width=150, x=170, y=100)

		btnVideo = tki.Button(self.root, text="Uploade your Video",command=self.selectFile)
		btnVideo.pack()
		btnVideo.place(height=50, width=310, x=10, y=200)
		

		btnPlay = tki.Button(self.root, text="Play",command=self.play)
		btnPlay.pack()
		btnPlay.place(height=50, width=150, x=10, y=300)

		btnPause = tki.Button(self.root, text="Pause",command=self.pause)
		btnPause.pack()
		btnPause.place(height=50, width=150, x=170, y=300)
		
		
		# create a button, that when pressed, will take the current
		# frame and save it to file
		btnE1 = tki.Button(self.root, text="Draw E1",command=self.drawE1)
		btnE1.pack()
		btnE1.place(height=50, width=150, x=10, y=400)
		btnS1 = tki.Button(self.root, text="Draw S1",command=self.drawS1)
		btnS1.pack()
		btnS1.place(height=50, width=150, x=170, y=400)
		
		btnE2 = tki.Button(self.root, text="Draw E2",command=self.drawE2)
		btnE2.pack()
		btnE2.place(height=50, width=150, x=10, y=500)
		btnS2 = tki.Button(self.root, text="Draw S2",command=self.drawS2)
		btnS2.pack()
		btnS2.place(height=50, width=150, x=170, y=500)

		btnE3 = tki.Button(self.root, text="Draw E3",command=self.drawE3)
		btnE3.pack()
		btnE3.place(height=50, width=150, x=10, y=600)
		btnS3 = tki.Button(self.root, text="Draw S3",command=self.drawS3)
		btnS3.pack()
		btnS3.place(height=50, width=150, x=170, y=600)

		btnE4 = tki.Button(self.root, text="Draw E4",command=self.drawE4)
		btnE4.pack()
		btnE4.place(height=50, width=150, x=10, y=700)
		btnS4 = tki.Button(self.root, text="Draw S4",command=self.drawS4)
		btnS4.pack()
		btnS4.place(height=50, width=150, x=170, y=700)

		btnE5 = tki.Button(self.root, text="Draw E5",command=self.drawE5)
		btnE5.pack()
		btnE5.place(height=50, width=150, x=10, y=800)
		btnS5 = tki.Button(self.root, text="Draw S5",command=self.drawS5)
		btnS5.pack()
		btnS5.place(height=50, width=150, x=170, y=800)
		

		# start a thread that constantly pools the video sensor for
		# the most recently read frame	
				
		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop, args=())
		self.thread.start()

		# set a callback to handle when the window is closed
		self.root.wm_title("Vehicles Tracking")
		self.root.geometry("1200x800")
		#self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

		with open('results.csv', 'w', newline='') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',',	quotechar='|', quoting=csv.QUOTE_MINIMAL)
			spamwriter.writerow(['Type', 'Entree', 'Sortie', 'Time'])



	def videoLoop(self):
		# DISCLAIMER:
		# I'm not a GUI developer, nor do I even pretend to be. This
		# try/except statement is a pretty ugly hack to get around
		# a RunTime error that Tkinter throws due to threading
		try:
			# keep looping over frames until we are instructed to stop
			while not self.stopEvent.is_set():
				if self.isRestart == True:
						self.vs = cv2.VideoCapture(self.videoPath)
						self.isRestart = False
				if self.isPlay:
					# grab the frame from the video stream and resize it to
					# have a maximum width of 800 pixels
					#self.sec=self.sec+self.frameRate
					#self.vs.set(cv2.CAP_PROP_POS_MSEC,self.sec*1000)
					self.frame = self.vs.read()
					self.frame = self.frame[1]
					self.frameCounting = self.frameCounting + 1
					if(self.frameCoumting == 900)
						# initialize OpenCV's special multi-object tracker
						self.trackers = cv2.MultiTracker_create()         
						self.trackingMatrix = [[]]
						self.countTracker = 0
					#fps = 30
					self.frame = imutils.resize(self.frame, width=800)
					(success, boxes) = self.trackers.update(self.frame)
					self.time = self.frameCounting / self.fps
					cv2.rectangle(self.frame, (self.e1[0], self.e1[1]), (self.e1[0] + self.e1[2], self.e1[1] + self.e1[3]), (255, 0, 0), 1)
					cv2.rectangle(self.frame, (self.s1[0], self.s1[1]), (self.s1[0] + self.s1[2], self.s1[1] + self.s1[3]), (0, 255, 0), 1)
					cv2.rectangle(self.frame, (self.e2[0], self.e2[1]), (self.e2[0] + self.e2[2], self.e2[1] + self.e2[3]), (255, 0, 0), 1)
					cv2.rectangle(self.frame, (self.s2[0], self.s2[1]), (self.s2[0] + self.s2[2], self.s2[1] + self.s2[3]), (0, 255, 0), 1)
					cv2.rectangle(self.frame, (self.e3[0], self.e3[1]), (self.e3[0] + self.e3[2], self.e3[1] + self.e3[3]), (255, 0, 0), 1)
					cv2.rectangle(self.frame, (self.s3[0], self.s3[1]), (self.s3[0] + self.s3[2], self.s3[1] + self.s3[3]), (0, 255, 0), 1)
					cv2.rectangle(self.frame, (self.e4[0], self.e4[1]), (self.e4[0] + self.e4[2], self.e4[1] + self.e4[3]), (255, 0, 0), 1)
					cv2.rectangle(self.frame, (self.s4[0], self.s4[1]), (self.s4[0] + self.s4[2], self.s4[1] + self.s4[3]), (0, 255, 0), 1)
					cv2.rectangle(self.frame, (self.e5[0], self.e5[1]), (self.e5[0] + self.e5[2], self.e5[1] + self.e5[3]), (255, 0, 0), 1)
					cv2.rectangle(self.frame, (self.s5[0], self.s5[1]), (self.s5[0] + self.s5[2], self.s5[1] + self.s5[3]), (0, 255, 0), 1)
					i=0;
					
					print(self.time)
					print(len(boxes))
					i=0;
					self.trackers = cv2.MultiTracker_create()
					
					for box in boxes:
						crossS = False
						(x, y, w, h) = [int(v) for v in box]
						#cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
						if (self.crossS1((x, y,x + w, y + h)) == True) :
							if(len(self.trackingMatrix[i])<=3) :
								self.trackingMatrix[i].append("s1")
								self.trackingMatrix[i].append(self.time)
								
								crossS = True
								
						if (self.crossS2((x, y,x + w, y + h)) == True) :
							if(len(self.trackingMatrix[i])<=3) :
								self.trackingMatrix[i].append("s2")
								self.trackingMatrix[i].append(self.time)
								
								crossS = True
								
						if (self.crossS3((x, y,x + w, y + h)) == True) :
							if(len(self.trackingMatrix[i])<=3) :
								self.trackingMatrix[i].append("s3")
								self.trackingMatrix[i].append(self.time)
								
								crossS = True
								
						if (self.crossS4((x, y,x + w, y + h)) == True) :
							if(len(self.trackingMatrix[i])<=3) :
								self.trackingMatrix[i].append("s4")
								self.trackingMatrix[i].append(self.time)
								
								crossS = True
								
						if (self.crossS5((x, y,x + w, y + h)) == True) :
							if(len(self.trackingMatrix[i])<=3) :
								self.trackingMatrix[i].append("s5")
								self.trackingMatrix[i].append(self.time)
								crossS = True
						
						if(crossS == True):
							with open('results.csv', 'a', newline='') as csvfile:
								spamwriter = csv.writer(csvfile, delimiter=',',	quotechar='|', quoting=csv.QUOTE_MINIMAL)
								spamwriter.writerow([self.trackingMatrix[i][0], self.trackingMatrix[i][1], self.trackingMatrix[i][2],self.trackingMatrix[i][3]])
							del self.trackingMatrix[i]
							self.countTracker = len(self.trackingMatrix)-1
							print("Sortie")
						else:
							tracker = self.OPENCV_OBJECT_TRACKERS[self.trackerType]()
							self.trackers.add(tracker, self.frame, (x, y, w, h))
							i=i+1

					if self.e1[3] > 0 :
						
						blob = Image(self.frame)
						
						results = self.net.detect(blob)

						self.postprocess(results)
						
						
					

						
					# OpenCV represents images in BGR order; however PIL
					# represents images in RGB order, so we need to swap
					# the channels, then convert to PIL and ImageTk format
					image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
					image = ImagePIL.fromarray(image)
					image = ImageTk.PhotoImage(image)
								# if the panel is not None, we need to initialize it
					if self.panel is None:
						self.panel = tki.Label(image=image)
						self.panel.image = image
						self.panel.pack()
						self.panel.place(x=350, y=200)
								# otherwise, simply update the panel
					else:
						self.panel.configure(image=image)
						self.panel.image = image
		except RuntimeError as e:
			print("[INFO] caught a RuntimeError")

	def selectFile(self):
		self.videoPath =  tki.filedialog.askopenfilename(initialdir = "",title = "Select file",filetypes = (("All files","*.*"),("mp4 files","*.mp4")))
		self.vs = cv2.VideoCapture(self.videoPath)
		self.fps = self.vs.get(cv2.CAP_PROP_FPS)
		#self.vs.set(cv2.CAP_PROP_POS_MSEC,self.sec*1000)
		self.frame = self.vs.read()
		self.frame = self.frame[1]
		self.frame = imutils.resize(self.frame, width=800)
		# OpenCV represents images in BGR order; however PIL
		# represents images in RGB order, so we need to swap
		# the channels, then convert to PIL and ImageTk format
		image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
		image = ImagePIL.fromarray(image)
		image = ImageTk.PhotoImage(image)
		# if the panel is not None, we need to initialize it
		if self.panel is None:
			self.panel = tki.Label(image=image)
			self.panel.image = image
			self.panel.pack()
			self.panel.place(x=350, y=200)
		# otherwise, simply update the panel
		else:
			self.panel.configure(image=image)
			self.panel.image = image		
	def restart(self):
		self.isRestart=True
		self.isPlay = False
		self.isPause = False
		self.e1 = (0, 0, 0, 0)
		self.e2 = (0, 0, 0, 0)
		self.e3 = (0, 0, 0, 0)
		self.s1 = (0, 0, 0, 0)
		self.s2 = (0, 0, 0, 0)
		self.s3 = (0, 0, 0, 0)

		# initialize OpenCV's special multi-object tracker
		self.trackers = cv2.MultiTracker_create() 
        
	def play(self):
		self.isRestart=False
		self.isPlay = True
		self.isPause = False
	def pause(self):
		self.isPause = True
		self.isPlay = False		
	def drawE1(self):
		self.e1 = cv2.selectROI("Select E1",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)
		print(self.e1)
	def drawS1(self):
		self.s1 = cv2.selectROI("Select S1",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)
		print(self.s1)
	def drawE2(self):
		self.e2 = cv2.selectROI("Select E2",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)
		print(self.e2)
	def drawS2(self):
		self.s2 = cv2.selectROI("Select S2",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)
		print(self.s2)
	def drawE3(self):
		self.e3 = cv2.selectROI("Select E3",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)
		print(self.e3)
	def drawS3(self):
		self.s3 = cv2.selectROI("Select S3",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)
		print(self.s3)
	
	def drawE4(self):
		self.e4 = cv2.selectROI("Select E4",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)
	def drawS4(self):
		self.s4 = cv2.selectROI("Select S4",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)

	def drawE5(self):
		self.e5 = cv2.selectROI("Select E5",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)
	def drawS5(self):
		self.s5 = cv2.selectROI("Select S5",imutils.resize(self.frame, width=800), fromCenter=False,showCrosshair=True)

	def crossE1(self,rect):
		if((self.e1[0]<=(rect[2]+rect[0])/2 and self.e1[0] + self.e1[2] >= (rect[2]+rect[0])/2) and (self.e1[1]<=(rect[3]+rect[1])/2 and self.e1[1] + self.e1[3] >= (rect[3]+rect[1])/2)):
			print("YES ! there is an overlap with e1")
			return True;
		else :
			return False;
	def crossS1(self,rect):
		if((self.s1[0]<=(rect[2]+rect[0])/2 and self.s1[0] + self.s1[2] >= (rect[2]+rect[0])/2) and (self.s1[1]<=(rect[3]+rect[1])/2 and self.s1[1] + self.s1[3] >= (rect[3]+rect[1])/2)):
			#print("YES ! there is an overlap with s1")
			return True;
		else :
			return False;	
	def crossE2(self,rect):
		if((self.e2[0]<=(rect[2]+rect[0])/2 and self.e2[0] + self.e2[2] >= (rect[2]+rect[0])/2) and (self.e2[1]<=(rect[3]+rect[1])/2 and self.e2[1] + self.e2[3] >= (rect[3]+rect[1])/2)):
			print("YES ! there is an overlap with e2")
			return True;
		else :
			return False;
	def crossS2(self,rect):
		if((self.s2[0]<=(rect[2]+rect[0])/2 and self.s2[0] + self.s2[2] >= (rect[2]+rect[0])/2) and (self.s2[1]<=(rect[3]+rect[1])/2 and self.s2[1] + self.s2[3] >= (rect[3]+rect[1])/2)):
			#print("YES ! there is an overlap with s2")
			return True;
		else :
			return False;	

	def crossE3(self,rect):
		if((self.e3[0]<=(rect[2]+rect[0])/2 and self.e3[0] + self.e3[2] >= (rect[2]+rect[0])/2) and (self.e3[1]<=(rect[3]+rect[1])/2 and self.e3[1] + self.e3[3] >= (rect[3]+rect[1])/2)):
			print("YES ! there is an overlap with e3")
			return True;
		else :
			return False;
	def crossS3(self,rect):
		if((self.s3[0]<=(rect[2]+rect[0])/2 and self.s3[0] + self.s3[2] >= (rect[2]+rect[0])/2) and (self.s3[1]<=(rect[3]+rect[1])/2 and self.s3[1] + self.s3[3] >= (rect[3]+rect[1])/2)):
			#print("YES ! there is an overlap with s3")
			return True;
		else :
			return False;


	def crossE4(self,rect):
		if((self.e4[0]<=(rect[2]+rect[0])/2 and self.e4[0] + self.e4[2] >= (rect[2]+rect[0])/2) and (self.e4[1]<=(rect[3]+rect[1])/2 and self.e4[1] + self.e4[3] >= (rect[3]+rect[1])/2)):
			print("YES ! there is an overlap with e4")
			return True;
		else :
			return False;
	def crossS4(self,rect):
		if((self.s4[0]<=(rect[2]+rect[0])/2 and self.s4[0] + self.s4[2] >= (rect[2]+rect[0])/2) and (self.s4[1]<=(rect[3]+rect[1])/2 and self.s4[1] + self.s4[3] >= (rect[3]+rect[1])/2)):
			#print("YES ! there is an overlap with s4")
			return True;
		else :
			return False;	

	def crossE5(self,rect):
		if((self.e5[0]<=(rect[2]+rect[0])/2 and self.e5[0] + self.e5[2] >= (rect[2]+rect[0])/2) and (self.e5[1]<=(rect[3]+rect[1])/2 and self.e5[1] + self.e5[3] >= (rect[3]+rect[1])/2)):
			print("YES ! there is an overlap with e5")
			return True;
		else :
			return False;
	def crossS5(self,rect):
		if((self.s5[0]<=(rect[2]+rect[0])/2 and self.s5[0] + self.s5[2] >= (rect[2]+rect[0])/2) and (self.s5[1]<=(rect[3]+rect[1])/2 and self.s5[1] + self.s5[3] >= (rect[3]+rect[1])/2)):
			#print("YES ! there is an overlap with s5")
			return True;
		else :
			return False;

	# Get the names of the output layers
	def getOutputsNames(self):
	    # Get the names of all the layers in the network
	    layersNames = self.net.getLayerNames()
	    # Get the names of the output layers, i.e. the layers with unconnected outputs
	    return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

	# Draw the predicted bounding box
	def drawPred(self, classId, conf, left, top, right, bottom):
		# Draw a bounding box.
		left = int(left)
		top = int(top)
		right =int(right)
		bottom = int(bottom)
		cv2.rectangle(self.frame, (int(left), int(top)), (int(right), int(bottom)), (255, 178, 50), 3) 
		# Get the label for the class name and its confidence
		rect = (left, top, right, bottom)
		label = classId
		if (self.crossE1(rect) == True) :
			tracker = self.OPENCV_OBJECT_TRACKERS[self.trackerType]()
			self.trackers.add(tracker, self.frame, (rect[0],rect[1] ,(rect[2]-rect[0]),(rect[3]-rect[1])))
			self.trackingMatrix[self.countTracker].append(label)
			self.trackingMatrix[self.countTracker].append("e1")
			
			self.trackingMatrix.append([])
			self.countTracker = self.countTracker + 1
		elif (self.crossE2(rect) == True) :
			tracker = self.OPENCV_OBJECT_TRACKERS[self.trackerType]()
			self.trackers.add(tracker, self.frame, (rect[0]+(rect[2]-rect[0])/3,rect[1] + (rect[3]-rect[1])/3 ,(rect[2]-rect[0])/3,(rect[3]-rect[1])/3))
			self.trackingMatrix[self.countTracker].append(label)
			self.trackingMatrix[self.countTracker].append("e2")
			self.trackingMatrix.append([])
			self.countTracker = self.countTracker + 1
		elif (self.crossE3(rect) == True) :
			tracker = self.OPENCV_OBJECT_TRACKERS[self.trackerType]()
			self.trackers.add(tracker, self.frame, (rect[0]+(rect[2]-rect[0])/3,rect[1] + (rect[3]-rect[1])/3 ,(rect[2]-rect[0])/3,(rect[3]-rect[1])/3))
			self.trackingMatrix[self.countTracker].append(label)
			self.trackingMatrix[self.countTracker].append("e3")
			self.trackingMatrix.append([])
			self.countTracker = self.countTracker + 1

		elif (self.crossE4(rect) == True) :
			tracker = self.OPENCV_OBJECT_TRACKERS[self.trackerType]()
			self.trackers.add(tracker, self.frame, (rect[0]+(rect[2]-rect[0])/3,rect[1] + (rect[3]-rect[1])/3 ,(rect[2]-rect[0])/3,(rect[3]-rect[1])/3))
			self.trackingMatrix[self.countTracker].append(label)
			self.trackingMatrix[self.countTracker].append("e4")
			self.trackingMatrix.append([])
			self.countTracker = self.countTracker + 1

		elif (self.crossE5(rect) == True) :
			tracker = self.OPENCV_OBJECT_TRACKERS[self.trackerType]()
			self.trackers.add(tracker, self.frame, (rect[0]+(rect[2]-rect[0])/3,rect[1] + (rect[3]-rect[1])/3 ,(rect[2]-rect[0])/3,(rect[3]-rect[1])/3))
			self.trackingMatrix[self.countTracker].append(label)
			self.trackingMatrix[self.countTracker].append("e5")
			self.trackingMatrix.append([])
			self.countTracker = self.countTracker + 1

			

# Remove the bounding boxes with low confidence using non-maxima suppression
	def postprocess(self, results):
		frameHeight = self.frame.shape[0]
		frameWidth = self.frame.shape[1]

		for cat, score, bounds in results:
			center_x, center_y, width, height = bounds
			left = int(center_x - width / 2)
			top = int(center_y - height / 2)
			self.drawPred(cat, score, left, top, left + width, top + height)
        		#cv.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
        		#cv.putText(frame,str(cat.decode("utf-8")),(int(x),int(y)),cv.FONT_HERSHEY_COMPLEX,1,(255,255,0))

	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		print("[INFO] closing...")
		#self.stopEvent.set()
		#self.vs.stop()
		self.root.quit()
