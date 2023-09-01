import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from loadDS import load_az_dataset, load_mnist_dataset
import cv2
from imutils import build_montages
from keras.models import load_model


plt.use("Agg")
EPOCHS = 10
# construct a plot that plots and saves the training history
images = []

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

labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, stratify=labels, random_state=42)

H = load_model('OCR_Resnet.keras')

N = np.arange(0, EPOCHS)
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(N, H.history["loss"], label="train_loss")
# plt.plot(N, H.history["val_loss"], label="val_loss")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.show()

output = ''

# randomly select a few testing characters
for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
  probs = H.predict(testX[np.newaxis, i])
  prediction = probs.argmax(axis=1)
  label = labelNames[prediction[0]]
  output+=label
  image = (testX[i] * 255).astype("uint8")
  color = (0, 255, 0)
  if prediction[0] != np.argmax(testY[i]):
    color = (0, 0, 255)
  image = cv2.merge([image] * 3)
  image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
  cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
		color, 2)
  images.append(image)
montage = build_montages(images, (96, 96), (7, 7))[0]
cv2.imshow('Result',montage)
cv2.waitKey(0)