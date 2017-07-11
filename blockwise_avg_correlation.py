def total_array(data):
        data=np.array([data])
        array=np.concatenate((array,data),axis=0)
      	global array
	return array
from sklearn import datasets, linear_model
import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
array=np.array([[0,0,0,0,0,0,0,0,0,0,0,0]])
targets=[]
with open('blockwise_feature_basic.txt','rb') as f:
    feature = [line.strip() for line in f]
targ=open("c_image_targets_basic.txt")
for y in targ.readlines():
        #if y.isdigit():
                targets.append(float(y))
for imag in range(0,1811):
	data=np.load(feature[imag])
	global data
	total_array(data);


avg_corr=0
alpha_para=0.15
for i in range(100):
        ############################# Train test split ###########################
        X_train, X_test, Y_train, Y_test = train_test_split(array,targets, test_size=0.2, random_state=i)
        reg=KernelRidge(alpha=alpha_para, kernel='rbf',degree=2) 
        reg.fit(X_train, Y_train)
        predicted=reg.predict(X_test)
        correlation=np.corrcoef(predicted,Y_test)
        avg_corr=avg_corr+correlation[0,1]
        print('*Lenear-correlation*:\\t',avg_corr/100)
        
