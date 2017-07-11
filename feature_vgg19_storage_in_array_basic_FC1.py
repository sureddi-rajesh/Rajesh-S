############ Make an array of all images features ##############################
def total_array(data):
        data=np.array(data)
        array=np.concatenate((array,data),axis=0)
      	global array
	return array
	
################## Imports #######################################
from sklearn import datasets, linear_model
import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import scipy
from sklearn.decomposition import PCA

##################### Load DATA #######################################
#array=np.array([[0,0,0,0,0,0,0,0,0,0,0,0]])
array=np.array([np.zeros(4096)])
print array.shape
targets=[]
with open('f_vgg19_fc1.txt','rb') as f:
    feature = [line.strip() for line in f]
targ=open("f_vgg19_image_targets_basic.txt")
for y in targ.readlines():
        #if y.isdigit():
                targets.append(float(y))
for imag in range(0,1811):
	data=np.load(feature[imag])
	global data
	total_array(data);
targets=np.array(targets)

######################### 4096 to 400 PCA ###############################
nf=400
pca=PCA(n_components=nf)
pca.fit(array)
new_array=pca.transform(array)
print new_array.shape

avg_corr=0
for i in range(100):
        ############################# Train test split ###########################
        X_train, X_test, Y_train, Y_test = train_test_split(new_array,targets, test_size=0.2, random_state=i)

        ############################## Training and Testing #######################
        reg=linear_model.LinearRegression()
        reg.fit(X_train, Y_train)
        predicted=reg.predict(X_test)
        correlation=np.corrcoef(predicted,Y_test)
        avg_corr=avg_corr+correlation[0,1]
        LNCC=scipy.stats.pearsonr(predicted,Y_test)
        SROCC=scipy.stats.pearsonr(predicted,Y_test)
        print('*LNCC::*','red',LNCC)
        print('*SROCC::*','blue',SROCC)
        print('*Lenear-correlation*:\\t',avg_corr/100)
        #alpha_para=0.1
