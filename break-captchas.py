from keras.models import load_model
import numpy as np 
import cv2
import os


def preprocess(image,width,height):

		(h,w)=image.shape[:2]
		if w>h:
			r=width/w
			image=cv2.resize(image,(width,int(r*h)))
		else:
			r=height/h
			image=cv2.resize(image,(int(r*w),height))
		
		padW=int((width-image.shape[1])/2.0)
		padH=int((height-image.shape[0])/2.0)
		image=cv2.copyMakeBorder(image,padH,padH,padW,padW,cv2.BORDER_REPLICATE)
		image=cv2.resize(image,(width,height))
		return image

imagepaths=list(os.listdir("downloads"))
imagepaths=["downloads/"+p for p in imagepaths]
imagepath=(np.random.choice(imagepaths,size=(1,),replace=False))[0]

print(imagepath)
model=load_model("captcha.hdf5")

image = cv2.imread(imagepath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20,cv2.BORDER_REPLICATE)
thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]



output = cv2.merge([gray]*3)
predictions = []

for c in cnts:

	(x, y, w, h) = cv2.boundingRect(c)
	roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

	roi = preprocess(roi, 28, 28)
	roi=(roi.reshape((1,28,28,1)))/255.0
	pred = model.predict(roi).argmax(axis=1)[0] + 1
	predictions.append(str(pred))

	cv2.rectangle(output, (x - 2, y - 2),(x + w + 4, y + h + 4), (0, 255, 0), 1)
	cv2.putText(output, str(pred), (x - 5, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
	cv2.imshow("output",output)
	cv2.waitKey(0)
