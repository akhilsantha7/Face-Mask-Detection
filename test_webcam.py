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
import cv2
import os

def MaskDetection(frame, face_detection, our_model):
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	face_detection.setInput(blob)
	detections = face_detection.forward()
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

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model ")
ap.add_argument("-m", "--model", type=str,
	default="Mask_detector.h5",
	help="path to trained mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="probability to filter weak detections")
args = vars(ap.parse_args())
print("loading face detector-----")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
face_detection = cv2.dnn.readNet(prototxtPath, weightsPath)
our_model = load_model(args["model"])

print("starting video stream-----")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	(locs, preds) = MaskDetection(frame, face_detection, our_model)
	i = 0
	for (box, pred) in zip(locs, preds):
    		
		i = i + 1
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	predictions.append(frame)
	if key == ord("q"):
		break

result_video = 'result.mp4'
imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
cv2.destroyAllWindows()
vs.stop()
