from PIL import Image
import glob

def catagorizeImage(fileName):
	pic=Image.open(fileName)
	pic.show()
	while True:
		catagory=input('Is this a cat or dog? (c or d)')
		if catagory=="c":
			pic.save('./testLabeled/cat.'+fileName[7:])
			break
		elif catagory=="d":
			pic.save('./testLabeled/dog.'+fileName[7:])
			break
		else: print('trouble parsing input, please try again.')
	pic.close()

def main():
	files=glob.glob('./test/*.jpg')
	for fileName in files:
		catagorizeImage(fileName)

if __name__ == '__main__':
	main()