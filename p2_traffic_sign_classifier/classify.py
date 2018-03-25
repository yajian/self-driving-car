import keras
import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from scipy.sparse import coo_matrix


def load_data():
	train_file = './train.p'
	test_file = './test.p'
	with open(train_file) as f:
		train=pickle.load(f)
	with open(test_file) as f:
		test = pickle.load(f)
	X_train,Y_train = train['data'],train['labels']
	X_test,Y_test = test['data'],test['labels']
	return X_train,Y_train,X_test,Y_test


def shuffle_data(X_train,Y_train,X_test,Y_test):
	X_train,Y_train = shuffle(X_train,Y_train,random_state = 0)
	X_test,Y_test = shuffle(X_test,Y_test,random_state=0)
	return X_train,Y_train,X_test,Y_test

def encode_label(Y_train,Y_test):
	Y_train_encoded = to_categorical(Y_train,num_classes = 43)
	Y_test_encode = to_categorical(Y_test,num_classes = 43)
	return Y_train_encoded,Y_test_encode

def normalize_data(X_train,X_test):
	X_train_normalize = (X_train - np.mean(X_train))/(np.max(X_train) - np.min(X_train))
	X_test_normalize = (X_test - np.mean(X_test))/(np.max(X_test) - np.min(X_test))
	return X_train_normalize,X_test_normalize

def split_data(X_train,Y_train,test_size = 0.3,random_state=0):
	x_train,x_dev,y_train,y_dev = train_test_split(X_train,Y_train,test_size = 0.3,random_state=0)
	return x_train,x_dev,y_train,y_dev

def build_model():
	model = Sequential()
	model.add(Conv2D(32,(3,3),input_shape = (32,32,3),activation = 'relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(512,activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(128,activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(43,activation='softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model



def main():
	X_train,Y_train,X_test,Y_test= load_data()
	X_train,Y_train,X_test,Y_test = shuffle_data(X_train,Y_train,X_test,Y_test)
	X_train,X_test = normalize_data(X_train,X_test)
	Y_train,Y_test = encode_label(Y_train,Y_test)
	X_train,x_dev,Y_train,y_dev = split_data(X_train,Y_train)
	model = build_model()
	model.fit(X_train,Y_train,validation_data=(x_dev,y_dev),epochs=40,batch_size=200,verbose =1)
	scores = model.evaluate(X_test,Y_test)
	print 'cnn error: %.2f%%'%(100-scores[1]*100)
	model.save('./model.h5')
	

if __name__ == '__main__':
	main()