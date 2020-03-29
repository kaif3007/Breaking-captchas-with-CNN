from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


import numpy as np 
import cv2
import os

from keras.models import Sequential

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense


from keras.optimizers import SGD
from keras import backend as K 

class LeNet:
	@staticmethod
	def build(width,height,depth,classes):
		model=Sequential()
		imageShape=(height,width,depth)

		if K.image_data_format()=="channels_first":
			imageShape=(depth,height,width)

		model.add(Conv2D(20,(5,5),padding="same",input_shape=imageShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
		model.add(Conv2D(50,(5,5),padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		return model



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


data=[]
labels=[]


for a in os.listdir("dataset"):
	for b in os.listdir("dataset/"+a):

		path='dataset/'+a+'/'+b
		image=cv2.imread(path)
		image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		image=preprocess(image,28,28)

		data.append(image)

		label=int(a)
		labels.append(label)


data=np.array(data,dtype="float")/255.0
data=data.reshape((data.shape[0],28,28,1))
labels=np.array(labels)

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.25)

lb=LabelBinarizer().fit(trainY)
trainY=lb.transform(trainY)
testY=lb.transform(testY)


model=LeNet.build(width=28,height=28,depth=1,classes=9)
sgd=SGD(lr=0.01)
model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])

H=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=32,epochs=15)

preds=model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1),preds.argmax(axis=1)))

model.save("captcha.hdf5")
