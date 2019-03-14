import glob
from PIL import Image


'''
Formats images to be smaller output size in order to reduce memory usage and increase processing speed
ARG: folderName    		Name of folder with images to reduce
ARG: destFolderTrain 	Name of the folder to output train images to
ARG: destFolderEval		Name of the folder to output Eval images to
ARG: size 				Size to resize images to 
'''
def formatImg(srcFolder, destFolder, size=100, genExtraData=False):
	#grab all jpg files in correct folder
	files=glob.glob('./'+srcFolder+'/*.jpg')

	nameStartIndex=files[0].rfind("/")+1

	for name in files:
		#open and convert pic to RGB
		pic=Image.open(name).convert("RGB")
		#remove folders to make naming easier
		name=name[nameStartIndex:]
		#resize pic to new size
		pic.thumbnail((size,size), Image.ANTIALIAS)
		#create another image with black backround
		back= Image.new('RGB',(size,size),(0,0,0,0))
		#paste pic into new backround image, centered
		back.paste(pic,((size-pic.size[0])//2,(size-pic.size[1])//2))
		#close orginal
		pic.close()
		#save new pic with black backround
		back.save('./'+destFolder+"/"+name[:4]+'fixed.0.'+name[4:])
		if genExtraData:
			#roate images and save again to provide more data
			back=back.rotate(90)
			back.save('./'+destFolder+"/"+name[:4]+'fixed.90.'+name[4:])
			back=back.rotate(90)
			back.save('./'+destFolder+"/"+name[:4]+'fixed.180.'+name[4:])
			back=back.rotate(90)
			back.save('./'+destFolder+"/"+name[:4]+'fixed.270.'+name[4:])

			#Mirror image for more data
			back.transpose(Image.FLIP_LEFT_RIGHT)
			#save and continue rotating
			back.save('./'+destFolder+"/"+name[:4]+'fixed.0.flip.'+name[4:])
			back=back.rotate(90)
			back.save('./'+destFolder+"/"+name[:4]+'fixed.90.flip.'+name[4:])
			back=back.rotate(90)
			back.save('./'+destFolder+"/"+name[:4]+'fixed.180.flip.'+name[4:])
			back=back.rotate(90)
			back.save('./'+destFolder+"/"+name[:4]+'fixed.270.flip.'+name[4:])
		#close back
		back.close()

def main():
	formatImg(srcFolder='train',destFolder='trainFormatted',genExtraData=False)


if __name__ == '__main__':
	main()