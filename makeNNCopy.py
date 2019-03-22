from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import initializers
from copy import deepcopy


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest')


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

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

def trainAndSave(model,epochs,name,image_size,trainDataLen,validDataLen):
	#hold on to best model to save after training
	bestModel=model
	bestModelLoss,bestModelAcc=1.0,0.0

	#moved these out of loop to solve mem alocation prob
	initBatchSize=16
	trainGen=trainGenerator(image_size,initBatchSize)
	validGen=validationGenerator(image_size,initBatchSize)

	for x in range(1,epochs+1):
		#print infor and adjust batch size
		print('MODEL: ',name,' CURRENT EPOCH:',x)
		batch_size=calBatchSize(x,epochs)
		#update generators only when batch size changes
		if batch_size!=initBatchSize:
			initBatchSize=batch_size
			trainGen=trainGenerator(image_size,batch_size)
			validGen=validationGenerator(image_size,batch_size)
		#fit model
		model.fit_generator(
		        trainGen,
		        steps_per_epoch=trainDataLen // batch_size,
		        epochs=1,
		        validation_data=validGen,
		        validation_steps=validDataLen // batch_size,
		        verbose=1,
		        max_queue_size=16)
		#cal loss and accuracy before comparing to previous best model
		loss,acc=model.evaluate_generator(validGen)
		if bestModelAcc<acc and bestModelLoss>loss:
			bestModel=deepcopy(model)
			bestModelLoss,bestModelAcc=loss,acc
	#save best model created
	bestModel.save_weights('./weights/weights_'+name+'_'+str(round(bestModelAcc,5))+'.h5')
	bestModel.save('./models/model_'+name+'_'+str(round(bestModelAcc,5))+'.dnn') 


def calBatchSize(epoch, totalEpochs):
	if epoch<=totalEpochs//6:
		return 16
	elif epoch<=(totalEpochs//6)*2:
		return 32
	elif epoch<=(totalEpochs//6)*3:
		return 64
	elif epoch<=(totalEpochs//6)*4:
		return 128
	elif epoch <=(totalEpochs//6)*5:
		return 256
	else:
		return 512

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

	trainAndSave(model,epochs,name,image_size,25000,1000)


#~95%
def model_original():

	dropout=0.2
	kernel_size=(3,3)
	pool_size=(2,2)
	image_size=300
	epochs=120
	name='og'

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
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name,image_size,25000,1000)

#added in two additional conv layers
def model_1():

	image_size=400
	dropout=0.2
	kernel_size=(3,3)
	pool_size=(2,2)
	name='model1'
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

	model.add(Conv2D(512, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(512, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
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
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name,image_size,25000,1000)


#increased Dropout rate (ineffective with dropout of .4, lowering to .3)
def model_2():

	image_size=300
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
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name,image_size,25000,1000)


#added another fully connected dense layer
def model_3():

	image_size=300
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
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name,image_size,25000,1000)

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
	while not modelStart(model_1):
		if input('Would you like to restart this model? (y or n) ')==n:
			break
	while not modelStart(model_2):
		if input('Would you like to restart this model? (y or n) ')==n:
			break
	while not modelStart(model_3):
		if input('Would you like to restart this model? (y or n) ')==n:
			break




if __name__ == '__main__':
	main()
