def block1_pool(input_im):
	from keras.applications.vgg16 import VGG16
	from keras.preprocessing import image
	from keras.applications.vgg16 import preprocess_input
	from keras.models import load_model
	from keras.models import Model
	import numpy as np
	import scipy.io
	import math
	base_model = VGG16(weights='imagenet')
	model=Model(input=base_model.input, output=base_model.get_layer('block1_pool').output)
	img_path = input_im
	#i = image.load_img(img_path, target_size=(224, 224,3))
	i = cv2.resize(img_path,(224,224))
	x = image.img_to_array(i)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	block1_pool_features=model.predict(x)
	layer1=block1_pool_features.flatten()
	receive1=estimate_aggd_params(MSCN(layer1))
	global receive1
	total_feature1=np.save('block_1_image_'+str(imag)+'.npy',receive1)
	return receive1

def block2_pool(input_im):
	from keras.applications.vgg16 import VGG16
	from keras.preprocessing import image
	from keras.applications.vgg16 import preprocess_input
	from keras.models import load_model
	from keras.models import Model
	
	import numpy as np
	import scipy.io
	import math
	base_model = VGG16(weights='imagenet')
	model=Model(input=base_model.input, output=base_model.get_layer('block2_pool').output)
	img_path = input_im
	#i = image.load_img(img_path, target_size=(3,224, 224))
	i = cv2.resize(img_path,(224,224))
	x = image.img_to_array(i)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	print x.shape
	block2_pool_features=model.predict(x)
	layer2=block2_pool_features.flatten()
	receive2=receive1+estimate_aggd_params(MSCN(layer2))
	global receive2
	total_feature2=np.save('block_12_image_'+str(imag)+'.npy',receive2)
	return receive2

def block3_pool(input_im):
	from keras.applications.vgg16 import VGG16
	from keras.preprocessing import image
	from keras.applications.vgg16 import preprocess_input
	from keras.models import load_model
	from keras.models import Model
	
	import numpy as np
	import scipy.io
	import math
	base_model = VGG16(weights='imagenet')
	model=Model(input=base_model.input, output=base_model.get_layer('block3_pool').output)
	img_path = input_im
	#i = image.load_img(img_path, target_size=(3,224, 224))
	i = cv2.resize(img_path,(224,224))
	x = image.img_to_array(i)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	print x.shape
	block3_pool_features=model.predict(x)
	layer3=block3_pool_features.flatten()
	receive3=receive2+estimate_aggd_params(MSCN(layer3))
	global receive3
	total_feature3=np.save('block_123_image_'+str(imag)+'.npy',receive3)
	return receive3

def block4_pool(input_im):
	from keras.applications.vgg16 import VGG16
	from keras.preprocessing import image
	from keras.applications.vgg16 import preprocess_input
	from keras.models import load_model
	from keras.models import Model
	
	import numpy as np
	import scipy.io
	import math
	base_model = VGG16(weights='imagenet')
	model=Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)
	img_path = input_im
	#i = image.load_img(img_path, target_size=(3,224, 224))
	i = cv2.resize(img_path,(224,224))
	x = image.img_to_array(i)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	print x.shape
	block4_pool_features=model.predict(x)
	layer4=block4_pool_features.flatten()
	receive4=receive3+estimate_aggd_params(MSCN(layer4))
	global receive4
	#scipy.io.savemat('block4_pool_features.mat',{'layer4':layer4},'5', oned_as='row')
	total_feature=np.save('image_'+str(imag)+'.npy',receive4)
	return total_feature
def estimate_aggd_params(x):
	x_left = x[x < 0]
	x_right = x[x >= 0]
	stddev_left = math.sqrt((1.0/(x_left.size - 1)) * np.sum(x_left ** 2))
	stddev_right = math.sqrt((1.0/(x_right.size - 1)) * np.sum(x_right ** 2))
	if stddev_right == 0:
	        return 1, 0, 0
	r_hat = np.mean(np.abs(x))**2 / np.mean(x**2)
	y_hat = stddev_left / stddev_right
	R_hat = r_hat * (y_hat**3 + 1) * (y_hat + 1) / ((y_hat**2 + 1) ** 2)
	alpha = generalized_gaussian_ratio_inverse(R_hat)
	beta_left = stddev_left * math.sqrt(gamma(3.0/alpha) / gamma(1.0/alpha))
	beta_right = stddev_right * math.sqrt(gamma(3.0/alpha) / gamma(1.0/alpha))
	return alpha, beta_left, beta_right
def generalized_gaussian_ratio_inverse(k):
    a1 = -0.535707356
    a2 = 1.168939911
    a3 = -0.1516189217
    b1 = 0.9694429
    b2 = 0.8727534
    b3 = 0.07350824
    c1 = 0.3655157
    c2 = 0.6723532
    c3 = 0.033834

    if k < 0.131246:
        return 2 * math.log(27.0/16.0) / math.log(3.0/(4*k**2))
    elif k < 0.448994:
        return (1/(2 * a1)) * (-a2 + math.sqrt(a2**2 - 4*a1*a3 + 4*a1*k))
    elif k < 0.671256:
        return (1/(2*b3*k)) * (b1 - b2*k - math.sqrt((b1 - b2*k)**2 - 4*b3*(k**2)))
    elif k < 0.75:
        return (1/(2*c3)) * (c2 - math.sqrt(c2**2 + 4*c3*math.log((3-4*k)/(4*c1))))
    else:
        print "warning: GGRF inverse of %f is not defined" %(k)
	return np.nan
def MSCN(x):
        u_x=np.mean(x)
        sigma_x=np.std(x)
        new_x=(x-u_x)/sigma_x
        return new_x


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
with open('HDR_database.txt','rb') as f:
    img = [line.strip() for line in f]
for imag in range(1811):
        global image
	input_im=cv2.imread(img[imag],1)
	block1_pool(input_im);
	block2_pool(input_im);
	block3_pool(input_im);
	block4_pool(input_im);

