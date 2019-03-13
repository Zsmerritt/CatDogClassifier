import glob, numpy as np, keras
from PIL import Image

def classifyImages(folder, model):
	#get names of all files in selcted folder
	for file in glob.glob('./'+folder+'/*.jpg'):
		#print(file[16:])
		pic=Image.open(file)
		#turn into numpy array
		datum=np.zeros((1,100,100,3),np.float16,'C')
		datum[0]=np.asarray(pic)
		#predict and save new image in correct folder
		prediction=model.predict(datum)
		print(prediction)
		if np.argmax(prediction) == 0:
			pic.save('./sorted/cats/'+file[16:])
		else:
			pic.save('./sorted/dogs/'+file[16:])
		pic.close()


def main():
	model=keras.models.load_model('trainedModel1.mlmodel')
	classifyImages('trainFormatted',model)

if __name__ == '__main__':
	main()