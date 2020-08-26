from tensorflow.keras.applications import Xception
from keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import imageio
import skimage
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model ")
ap.add_argument("-m", "--model", type=str,
	default="Mask_detector.h5",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

def MaskDetection(frame, face_det, our_model):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	face_det.setInput(blob)
	detections = face_det.forward()
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = our_model.predict(faces, batch_size=32)
	return (locs, preds)



print("loading face detector model-----")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
face_det = cv2.dnn.readNet(prototxtPath, weightsPath)

print("loading face mask detector model------")
our_model = load_model(args["model"])

print("starting video stream------")

vidcap = cv2.VideoCapture('video.mp4')
success,image = vidcap.read()
count = 0
print(success)
predictions = []

while success:
	success,frame = vidcap.read()
	print('Read a new frame: ', success)
	if frame is None:
    		break

	frame = imutils.resize(frame, width=400)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	
	(locs, preds) = MaskDetection(frame, face_det, our_model)

	
	i = 0
	
	for (box, pred) in zip(locs, preds):
    		
		i = i + 1
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	predictions.append(frame)
	count += 1

result_video = 'video_result.mp4'
imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=25)

cv2.destroyAllWindows()
