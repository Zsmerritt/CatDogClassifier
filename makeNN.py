import numpy as np, os, keras, glob, sys
from PIL import Image

def makeNetwork():

	poolSize=(2,2)
	dropout=0.2

	model=keras.models.Sequential()
	#convolutional layers
	#model.add(keras.layers.Conv2D(32,kernel_size=(3,3),padding='same',activation=('relu'),input_shape=(100,100,3)))
	model.add(keras.layers.Conv2D(128,kernel_size=(3,3),padding='same',activation=('relu')))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D(pool_size=(poolSize)))
	model.add(keras.layers.Dropout(dropout))

	model.add(keras.layers.Conv2D(256,kernel_size=(3,3),padding='same',activation=('relu')))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D(pool_size=(poolSize)))
	model.add(keras.layers.Dropout(dropout))

	model.add(keras.layers.Conv2D(512,kernel_size=(5,5),padding='same',activation=('relu')))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D(pool_size=(poolSize)))
	model.add(keras.layers.Dropout(dropout))

	model.add(keras.layers.Conv2D(1024,kernel_size=(5,5),padding='same',activation=('relu')))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D(pool_size=(poolSize)))
	model.add(keras.layers.Dropout(dropout))

	model.add(keras.layers.Conv2D(512,kernel_size=(5,5),padding='same',activation=('relu')))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D(pool_size=(poolSize)))
	model.add(keras.layers.Dropout(dropout))

	model.add(keras.layers.Conv2D(256,kernel_size=(5,5),padding='same',activation=('relu')))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D(pool_size=(poolSize)))
	model.add(keras.layers.Dropout(dropout))

	#model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	#model.add(keras.layers.Dropout(0.5))

	#model.add(keras.layers.Conv2D(2048,kernel_size=(5,5),padding='same',activation=('relu')))
	#model.add(keras.layers.Conv2D(4096,kernel_size=(5,5),padding='same',activation=('relu')))
	#model.add(keras.layers.Conv2D(4096,kernel_size=(5,5),padding='same',activation=('relu')))

	#flatten
	model.add(keras.layers.Flatten())
	#Dense layers
	model.add(keras.layers.Dense(4096))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Activation('relu'))
	model.add(keras.layers.Dropout(dropout))

	model.add(keras.layers.Dense(2048))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Activation('relu'))
	model.add(keras.layers.Dropout(dropout))

	model.add(keras.layers.Dense(1024))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Activation('relu'))
	model.add(keras.layers.Dropout(dropout))

	model.add(keras.layers.Dense(512))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Activation('relu'))
	model.add(keras.layers.Dropout(dropout))

	model.add(keras.layers.Dense(32))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Activation('relu'))
	model.add(keras.layers.Dropout(dropout))

	#final softmax layer
	model.add(keras.layers.Dense(2,activation='softmax'))
	#compile
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.adam(),metrics=['accuracy'])
	return model

def testModel(data, labels, model):
	correct=0
	datum=np.zeros((1,100,100,3),np.float16,'C')
	for x in range(len(data)):
		datum[0]=data[x]
		prediction=model.predict(datum)
		if np.argmax(prediction) == 0:
			#cats
			if labels[x][0]==1:correct+=1
		else:
			#dogs
			if labels[x][0]==0:correct+=1
	return correct/len(labels)

def importDataset(folderName):
	#grab files and initilize new arrays
	files=glob.glob('./'+folderName+'/*.jpg')
	data=np.zeros(((len(files)),100,100,3),np.float16,'C')
	labels=np.zeros((len(files),2))
	#for each file...
	letterIndex=files[0].rfind("/")+1

	for x in range(len(files)):
		try:
		#open, convert to array, then close
			pic=Image.open(files[x])
			picture=np.asarray(pic)
			pic.close()
			#set to the correct list
			data[x]=picture
		except: print('Error occured. Couldnt open picture:',files[x])
		if files[x][letterIndex] == "d": labels[x]=[0,1]
		else: labels[x]=[1,0]
	return(data,labels)


def shuffleData(data,labels):
	perm=np.random.permutation(data.shape[0])
	return(data[perm],labels[perm])


def adjustBatchSize(acc):
	if acc<=.5:
		return 128
	elif acc<=.75:
		return 256
	elif acc<=.85:
		return 512
	elif acc<=.90:
		return 1024
	else:
		return 2048


def main():
	#make the network
	print('making network')
	network=makeNetwork()
	print('Done making network')


	#try to get both data sets
	print('importing data')
	try:
		data,labels=importDataset('trainFormatted')
		testData,testLabels=importDataset('testFormatted')
	except FileNotFoundError as e: print(e,'Could not find data. Please try again')
	print('Done importing data')


	#Continue executing epochs until error starts increasing
	print('training network')
	prevAcc=0.0
	acc=0.4
	counter=0
	batchSize=128
	while acc>=prevAcc:
		#switch error and model in prep for new epoch
		prevAcc=acc
		prevModel=network
		counter+=1
		#execute epoch
		network.fit(x=data,y=labels,batch_size=batchSize,epochs=1,verbose=1)
		#shuffle data to normalize & remove bias
		data,labels=shuffleData(data,labels)
		#calculate error of new model
		acc=round(testModel(testData,testLabels,network)*10,2)
		#adjust batch size according to accuracy
		batchSize=adjustBatchSize(acc)
		print('Accuracy before epoch: ',prevAcc,'Accuracy after epoch: ',acc)
		network.save('trainedModel'+str(counter)+'.dnn',include_optimizer=False)
	print('Finished!')

if __name__ == '__main__':
	main()





