#updated memory allocation, hopfully fixes epoch 41 mem allocation
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import initializers
from copy import deepcopy
import gc
import dataGen
from tqdm import tqdm
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

train_transform_map = dataGen.get_transform_map(
	data_folder='./data/train/',
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest')

valid_transform_map = dataGen.get_transform_map(data_folder='./data/testLabeled/', rescale=1./255)

train_datagen = ImageDataGenerator(
		rescale=1./255,
		horizontal_flip=True,
		vertical_flip=True,
		fill_mode='nearest',
		brightness_range=(0.0,1.5),
		shear_range=0.2,
    	zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)



'''
print('generating validation data')
#train=dataGen.image_processor(transform_map=train_transform_map,target_size=target_size,image_multiplier=2)
print('finished processing data')

'''
# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
def trainGenerator(size, batch):
	return train_datagen.flow_from_directory(
        './data/train/',  # this is the target directory
        target_size=(size, size),  # all images will be resized to 150x150
        batch_size=batch,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

def validationGenerator(size, batch):
# this is a similar generator, for validation data
	return test_datagen.flow_from_directory(
        './data/testLabeled/',
        target_size=(size, size),
        batch_size=batch,
        class_mode='binary')

def shuffleData(data_dict):
	perm=np.random.permutation(data_dict['data'].shape[0])
	data_dict['data'],data_dict['labels']=data_dict['data'][perm],data_dict['labels'][perm]

#using generator
def trainAndSaveGenerator(model,epochs,name,target_size,batch_size,model_save_filepath):
	#hold on to best model to save after training
	trainGen=trainGenerator(target_size,batch_size)
	validGen=validationGenerator(target_size,batch_size)
	#print info and start epoch
	hist=model.fit_generator(
			trainGen,
			steps_per_epoch=25000 // batch_size,
			#steps_per_epoch=trainDataLenP // batch_size,
			epochs=epochs,
			validation_data=validGen,
			validation_steps=1000 // batch_size,
			#validation_steps=validDataLenP // batch_size,
			verbose=1,
			max_queue_size=16,
			#use_multiprocessing=True,
			#workers=2,
			callbacks=[
				EarlyStopping(patience=6, monitor='val_acc'),
				ReduceLROnPlateau(patience=3,factor=0.4,min_lr=0.001),
				ModelCheckpoint(model_save_filepath, monitor='val_acc', save_best_only=False)

			])

#trains on batch
def trainAndSave(model,epochs,name,target_size):
	#hold on to best model to save after training
	bestModel=model
	bestModelLoss,bestModelAcc=1.0,0.0


	try:
		for x in range(0,epochs):
			#update batch_size 
			batch_size=calBatchSize(x+1,epochs)
			steps_per_epoch_train=25000//batch_size
			epoch_desc='MODEL: '+str(name)+'  CURRENT EPOCH: '+str(x+1)+"/"+str(epochs)+'  BATCH SIZE: '+str(batch_size)
			for y in tqdm(range(steps_per_epoch_train), desc=epoch_desc):

				train=dataGen.image_processor_batch(transform_map=train_transform_map,target_size=target_size,batch_size=batch_size)
				model.train_on_batch(
			        x=train['data'],
			        y=train['labels'])
			'''
			#cal loss and accuracy before comparing to previous best model
			acc = model.evaluate(
							x=valid['data'],
							y=valid['labels'],
							batch_size=batch_size,
							verbose=1)
							#['val_acc'][0],hist.history['val_loss'][0]
			'''
			acc=test_model_accuracy(model=model,transform_map=valid_transform_map,target_size=target_size,batch_size=batch_size)
			print("Model Validation Accuracy: ",acc)
			if bestModelAcc<acc:
				bestModel=deepcopy(model)
				bestModelAcc=acc
		#save best model created
		bestModel.save_weights('./weights/weights_'+name+'_'+str(round(bestModelAcc,5))+'.h5')
		bestModel.save('./models/model_'+name+'_'+str(round(bestModelAcc,5))+'.dnn') 
	except KeyboardInterrupt as e:
		print('Saving best model generated so far')
		bestModel.save_weights('./weights/weights_'+name+'_'+str(round(bestModelAcc,5))+'.h5')
		bestModel.save('./models/model_'+name+'_'+str(round(bestModelAcc,5))+'.dnn') 
		raise KeyboardInterrupt

def test_model_accuracy(model,transform_map,target_size,batch_size):
	correct=0
	for x in tqdm(range(999),desc='Evaluating Model'):
		valid=dataGen.image_processor_batch(transform_map=transform_map,target_size=target_size,batch_size=1)
		prediction=round(model.predict(valid['data'].reshape(1, target_size[0], target_size[1], 3))[0][0])
		if prediction== valid['labels'][0]:correct+=1
	return correct/999



def calBatchSize(epoch, totalEpochs):
	if epoch<=totalEpochs//6:
		return 32
	elif epoch<=(totalEpochs//6)*2:
		return 64
	elif epoch<=(totalEpochs//6)*3:
		return 128
	elif epoch<=(totalEpochs//6)*4:
		return 256
	elif epoch <=(totalEpochs//6)*5:
		return 512
	else:
		return 1024

def modelOld():

	dropout=0.5
	kernel_size=(3,3)
	pool_size=(2,2)
	image_size=150
	epochs=50
	name='old'

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, input_shape=(image_size, image_size, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Conv2D(32, kernel_size=kernel_size))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Conv2D(64, kernel_size=kernel_size))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name)


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
	#model.add(MaxPooling2D(pool_size=pool_size))
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
	#model.add(MaxPooling2D(pool_size=pool_size))
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

	model.add(Conv2D(1024, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 15
	#c = 3/15 = 0.2

	model.add(Conv2D(1024, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 17
	#c = 3/17 = 0.17 
	#this is perfect, it should get as close to 1/6 = 0.16 without going below
	#it might be worth putting in a stride len > 1, which would increase Receptive Field Size, and therefore allow the model to see more of the big picture
	#this may also mean removing some of the deeper convolutional layers. This could be rectified by increasing kernal size

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


#added in two additional conv layers
def model_1():

	image_size=256
	dropout=0.2
	kernel_size=(3,3)
	pool_size=(2,2)
	name='model1'
	epochs=120
	stride=(2,2)

	#receptive field size = prevLayerRCF + (K-1) * jumpSize
	#featOut = ceil((featIn + 2*padding - K)/strideLen)+1
	#jumpOut = (featInit-featOut)/featOut-1  OR  stride*JumpIn

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, padding='same', stride=stride, input_shape=(image_size, image_size, 3), kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS= 1 + 2*1= 3
	#c=3/3=1

	model.add(Conv2D(32, kernel_size=kernel_size, padding='same', stride=stride, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS= 3+2^2 = 7
	#c=3/7

	model.add(Conv2D(64, kernel_size=kernel_size, padding='same', stride=stride, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS= 7 + 2^3=15
	#c=3/15=0.2

	model.add(Conv2D(64, kernel_size=kernel_size, padding='same', stride=stride, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 15 + 2^4=31

	model.add(Conv2D(128, kernel_size=kernel_size, padding='same', stride=stride, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 31 +2^5=63

	model.add(Conv2D(128, kernel_size=kernel_size, padding='same', stride=stride, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 63 + 2^6 = 127

	model.add(Conv2D(256, kernel_size=kernel_size, padding='same', stride=stride, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 127 + 2^7 = 255

	'''
	model.add(Conv2D(256, kernel_size=kernel_size, padding='same', stride=stride, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 255 + 2^8 = 511
	'''


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

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name,target_size=(image_size,image_size))


#increased Dropout rate (ineffective with dropout of .4, lowering to .3)
def model_2():

	image_size=400
	#increased dropout from 0.2 to 0.3
	dropout=0.3
	kernel_size=(3,3)
	pool_size=(2,2)
	name='model2'
	epochs=120

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, padding='same', input_shape=(image_size, image_size, 3), kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(32, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(128, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(128, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(256, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(256, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

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

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name)


#added another fully connected dense layer
def model_3():

	image_size=400
	dropout=0.2
	kernel_size=(3,3)
	pool_size=(2,2)
	name='model3'
	epochs=120

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, padding='same', input_shape=(image_size, image_size, 3), kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(32, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(128, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(128, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(256, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(256, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	#added extra dense 512 layer on top
	model.add(Dense(512, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

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

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name)

#method allows for restarting models which are trainging poorly
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
	'''
	while not modelStart(model_1):
		if input('Would you like to restart this model? (y or n) ')==n:
			break
	while not modelStart(model_2):
		if input('Would you like to restart this model? (y or n) ')==n:
			break
	while not modelStart(model_3):
		if input('Would you like to restart this model? (y or n) ')==n:
			break
	'''




if __name__ == '__main__':
	main()
