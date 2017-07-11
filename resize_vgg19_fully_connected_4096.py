def fc1_pool(input_im):
	from keras.applications.vgg19 import VGG19
	from keras.preprocessing import image
	from keras.applications.vgg19 import preprocess_input
	from keras.models import load_model
	from keras.models import Model
	import numpy as np
	import scipy.io
	import math
	base_model = VGG19(weights='imagenet')
	model=Model(input=base_model.input, output=base_model.get_layer('fc1').output)
	img_path = input_im
	#print img_path.shape
	#i = image.load_img(img_path, target_size=(224, 224,3))
	i = cv2.resize(img_path,(224,224))
	x = image.img_to_array(i)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	#print x.shape
	fc1_features=model.predict(x)
	total_feature=np.save('vgg19_fc1_image_'+str(imag)+'.npy',fc1_features)
	#scipy.io.savemat('block1_pool_features.mat',{'layer1':layer1},'5', oned_as='row')
	#np.save('block1_pool_features.npy',layer1)
	return total_feature



from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.models import Model
import PIL
import numpy as np
import os, sys
import cv2
import math
import numpy.linalg
from scipy.special import gamma
from scipy.ndimage.filters import gaussian_filter
import scipy.misc
import scipy.io
import colorsys
#text_file=open("filenames.txt","r")
#lines=text_file.readlines()
#text_file.close()
#image=0
with open('HDR_database.txt','rb') as f:
    img = [line.strip() for line in f]
for imag in range(1811):
        global image
        input_im=cv2.imread(img[imag],1)
        fc1_pool(input_im);
