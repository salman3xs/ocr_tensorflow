from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import tensorflow as tf
from keras.models import load_model

loaded_model = load_model('OCR_Resnet.keras') 

image = cv2.imread('images/ocr4.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]
chars = []
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
		roi = gray[y:y + h, x:x + w]
		thresh = cv2.threshold(roi, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		(tH, tW) = thresh.shape
		if tW > tH:
			thresh = imutils.resize(thresh, width=32)
		else:
			thresh = imutils.resize(thresh, height=32)
		(tH, tW) = thresh.shape
		dX = int(max(0, 32 - tW) / 2.0)
		dY = int(max(0, 32 - tH) / 2.0)
		padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
			left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
			value=(0, 0, 0))
		padded = cv2.resize(padded, (32, 32))
		padded = padded.astype("float32") / 255.0
		padded = np.expand_dims(padded, axis=-1)
		chars.append((padded, (x, y, w, h)))

# final_image=np.expand_dims(padded,axis=2)
# print(final_image.shape)
# cv2.imshow('Intial Image',edged)
# cv2.waitKey(0)

boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")
preds = loaded_model.predict(chars)
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

output=""
for (pred, (x, y, w, h)) in zip(preds, boxes):
  i = np.argmax(pred)
  prob = pred[i]
  label = labelNames[i]
  output+=label
  print("[INFO] {} - {:.2f}%".format(label, prob * 100))
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
  cv2.putText(image, label, (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
img = cv2.resize(image,(960,540))
cv2.imshow('Result',img)
cv2.waitKey(0)