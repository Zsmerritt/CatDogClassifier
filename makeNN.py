import numpy as np, os, keras, glob, sys
from PIL import Image
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import initializers

train_datagen = ImageDataGenerator(
		rescale=1./255,
		horizontal_flip=True,
		vertical_flip=True,
		fill_mode='nearest',
		brightness_range=(0.0,1.5),
		shear_range=0.2,
    	zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

def trainGenerator(size, batch):
	return train_datagen.flow_from_directory(
        './data/train/',  # this is the target directory
        target_size=size,  # all images will be resized to 150x150
        batch_size=batch,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

def validationGenerator(size, batch):
# this is a similar generator, for validation data
	return test_datagen.flow_from_directory(
        './data/testLabeled/',
        target_size=size,
        batch_size=batch,
        class_mode='binary')


def trainAndSaveGenerator(model,epochs,name,target_size,batch_size,model_save_filepath):
	#print info and start epoch
	model.fit_generator(
		trainGenerator(target_size,batch_size),
		steps_per_epoch=25000 // batch_size,
		epochs=epochs,
		validation_data=validationGenerator(target_size,batch_size),
		validation_steps=1000 // batch_size,
		verbose=1,
		max_queue_size=16,
		use_multiprocessing=True,
		workers=8,
		callbacks=[
			EarlyStopping(patience=10, monitor='val_acc'),
			ModelCheckpoint(model_save_filepath, monitor='val_acc', save_best_only=False),
			ReduceLROnPlateau(patience=3,factor=0.4,min_lr=0.001)
		])

#~95%
def model_original():

	dropout=0.2
	kernel_size=(3,3)
	pool_size=(2,2)
	image_size=256
	target_size=(256,256)
	epochs=120
	name='model-1'
	batch_size=64
	filepath='./models/'+name+'.{epoch:02d}-{val_acc:.2f}.hdf5'


	#receptive field size = prevLayerRCF + (K-1) * jumpSize
	#featOut = ceil((featIn + 2*padding - K)/strideLen)+1
	#jumpOut = (featInit-featOut)/featOut-1  OR  stride*JumpIn

	model = Sequential()
	model.add(Conv2D(128, kernel_size=kernel_size, padding='same', input_shape=(image_size, image_size, 3), kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#receptive field size = 1 + (3-1) * 1 = 3
	#real Filter size = 3
	#c = 1

	model.add(Conv2D(128, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 3 + 2 * 1 = 5
	#FilterSize = 3
	#c = 3 / 5 = 0.6

	model.add(Conv2D(256, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 5+2=7
	#c = 3 / 7 = 0.42

	model.add(Conv2D(256, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS= 9
	#c = 3 / 9 = 0.33

	model.add(Conv2D(512, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 11
	#c = 3/11 = 0.2727

	model.add(Conv2D(512, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 13
	#c = 3/13 = 0.23

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(256, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(128, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(64, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(32, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	#it might be good to try freezing all convolution layers or all layers except last 32 Dense and training that specific layer to be more accurate.

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	trainAndSaveGenerator(model,epochs,name,target_size,batch_size,filepath)

def modelStart(modelName):
	try:
		modelName()
		return True
	except KeyboardInterrupt as e:
		print('KeyboardInterrupt detected, ending training')
		return False


def main():
	while not modelStart(model_original):
		if input('Would you like to restart this model? (y or n) ')==n:
			break

if __name__ == '__main__':
	main()





