from sklearn.preprocessing import LabelBinarizer
from loadDS import load_az_dataset, load_mnist_dataset
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from resNet import ResNet

SGD = tf.keras.optimizers.SGD
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
print(tf.test.is_built_with_cuda())
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print("Num GPUs Available: ", tf.config.list_physical_devices('GPU')[0])
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# print(tf.config.list_physical_devices('GPU'))
# gpu_device = tf.config.list_physical_devices('GPU')[0]
# session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1},
#                                          log_device_placement=True))

(digitsData, digitsLabels) = load_mnist_dataset()

(azData, azLabels) = load_az_dataset('ds/A_Z Handwritten Data.csv')
print("data Imported")

azLabels += 10

data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])

data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

data = np.expand_dims(data, axis=-1)
data /= 255.0
print("data Mapped")
le = LabelBinarizer()
labels = le.fit_transform(labels)

counts = labels.sum(axis=0)

classTotals = labels.sum(axis=0)
classWeight = {}

for i in range(0, len(classTotals)):
  classWeight[i] = classTotals.max() / classTotals[i]

(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, stratify=labels, random_state=42)

aug = ImageDataGenerator(
rotation_range=10,
zoom_range=0.05,
width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.15,
horizontal_flip=False,
fill_mode="nearest")

print("data Augmented")

EPOCHS = 10
INIT_LR = 1e-1
BS = 128

opt = SGD(learning_rate=INIT_LR)

model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
(64, 64, 128, 256), reg=0.0005)

print('RsNet Build')
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
print('model complied')
# model.load_weights('/gdrive/MyDrive/Colab Notebooks/OCR/Models/OCR_Resnet')


H = model.fit(
aug.flow(trainX, trainY, batch_size=BS),
validation_data=(testX, testY),
steps_per_epoch=len(trainX) // BS,epochs=EPOCHS,
class_weight=classWeight,
verbose=1)
print('model fitted')

labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

print("Making Predictions")

predictions = model.predict(testX, batch_size=BS)

print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

model.save('OCR_Resnet.keras')
print('model Saved')