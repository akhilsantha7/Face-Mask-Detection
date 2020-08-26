from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception
from keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--input_data", required=True,
	help="path to  dataset")
args = vars(ap.parse_args())

EPOCHS = 25
batch_size = 32
learning_rate = 0.001
print("loading images...")
image_paths = list(paths.list_images(args["input_data"]))
data = []
labels = []

for path in image_paths:
	label = path.split(os.path.sep)[-2]
	image = load_img(path, target_size=(299, 299))
	image = img_to_array(image)
	image = preprocess_input(image)	
	data.append(image)
	labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

#  one-hot encoding 
enc = LabelBinarizer()
labels = enc.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)

aug_data = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the Xception network, Fc layers are not taken

basemodel = Xception(weights='imagenet',
        include_top=False, input_tensor=Input(shape=(299, 299, 3)))


new_model = basemodel.output
new_model = AveragePooling2D(pool_size=(7, 7))(new_model)
new_model = Flatten(name="flatten")(new_model)
new_model = Dense(128, activation="relu")(new_model)
new_model = Dropout(0.5)(new_model)
new_model = Dense(2, activation="softmax")(new_model)


model = Model(inputs=basemodel.input, outputs=new_model)

for layer in basemodel.layers:
	layer.trainable = False

opt = Adam(lr=learning_rate, decay=learning_rate / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("Training------")
H = model.fit(
	aug_data.flow(trainX, trainY, batch_size=batch_size),
	steps_per_epoch=len(trainX) // batch_size,
	validation_data=(testX, testY),
	validation_steps=len(testX) // batch_size,
	epochs=EPOCHS)

# Test part
print("Evaluating-------")
predict_val = model.predict(testX, batch_size=batch_size)
predict_val = np.argmax(predict_val, axis=1)
print(classification_report(testY.argmax(axis=1), predict_val,
	target_names=enc.classes_))
print("Saving the model")
model.save('Mask_detector.h5')

