from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import initializers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session



def model1():
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(150, 150, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, kernel_size=(3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, kernel_size=(3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	batch_size = 16

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
	train_generator = train_datagen.flow_from_directory(
	        './data/train/',  # this is the target directory
	        target_size=(150, 150),  # all images will be resized to 150x150
	        batch_size=batch_size,
	        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

	# this is a similar generator, for validation data
	validation_generator = test_datagen.flow_from_directory(
	        './data/testLabeled/',
	        target_size=(150, 150),
	        batch_size=batch_size,
	        class_mode='binary')


	model.fit_generator(
	        train_generator,
	        steps_per_epoch=25000 // batch_size,
	        epochs=50,
	        validation_data=validation_generator,
	        validation_steps=1000 // batch_size)
	model.save_weights('first_try1.h5')
	model.save('model1.dnn') 

def model2():

	image_size=150
	dropout=0.5
	kernel_size=(3,3)
	pool_size=(2,2)

	model = Sequential()
	model.add(Conv2D(64, kernel_size=kernel_size, input_shape=(image_size, image_size, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Conv2D(64, kernel_size=kernel_size))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))


	model.add(Conv2D(128, kernel_size=kernel_size))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Conv2D(128, kernel_size=kernel_size))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))


	model.add(Conv2D(256, kernel_size=kernel_size))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Conv2D(256, kernel_size=kernel_size))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))


	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dense(128, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dense(64, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dense(32, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	batch_size = 16

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
	train_generator = train_datagen.flow_from_directory(
	        './data/train/',  # this is the target directory
	        target_size=(image_size, image_size),  # all images will be resized to 150x150
	        batch_size=batch_size,
	        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

	# this is a similar generator, for validation data
	validation_generator = test_datagen.flow_from_directory(
	        './data/testLabeled/',
	        target_size=(image_size, image_size),
	        batch_size=batch_size,
	        class_mode='binary')


	model.fit_generator(
	        train_generator,
	        steps_per_epoch=25000 // batch_size,
	        epochs=50,
	        validation_data=validation_generator,
	        validation_steps=1000 // batch_size)
	model.save_weights('first_try2.h5')
	model.save('model2.dnn') 


def main():
	model2()
	model1()


if __name__ == '__main__':
	main()
