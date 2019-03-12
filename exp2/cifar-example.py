#from cleverhans.attacks import FastGradientMethod
#from cleverhans.utils_keras import KerasModelWrapper
#from cleverhans.attacks import SaliencyMapMethod

import numpy as np
import numpy as np
from keras.models import *
from keras import backend
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.applications.vgg16 import VGG16

import sys
sys.path.insert(0, '../')
from utils import *
from sbfl import *
#from sbfl_copy import *
import sys

img_rows, img_cols, img_channels = 32, 32, 3
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
x_test = x_test.astype('float32')
x_test /= 255

idx=np.random.randint(0,len(x_test))
x=np.array([x_test[idx]])
#print ('x data shape:', x.shape)

model=load_model('../saved_models/cifar10_complicated.h5')

for idx in range(len(x_test)-1, -1, -1):
  #idx=3 #np.random.randint(0,len(x_test))

  x=np.array([x_test[idx]])
  y=np.argsort(model.predict(x))[0][-1:]
  if y[0]!=y_test[idx][0]: continue
  
  adv_xs=[]
  adv_ys=[]
  num_advs=0
  
  for c in range(0, 1000):
    sp=x.shape
    sigma=0.1 #np.random.uniform(0,0.1)
    noise=np.random.normal(size=sp, scale=sigma)
    adv_x=x+noise
    adv_x=np.clip(adv_x, a_min=0., a_max=1.)
    adv_y=np.argsort(model.predict(adv_x))[0][-1:]
    if adv_y[0]!=y[0]:
      num_advs+=1
    adv_xs.append(adv_x[0])
    adv_ys.append(adv_y[0])
  
  print ('###', num_advs)
  if num_advs==0: continue #sys.exit(0)
  if num_advs>900: continue

  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.05, attack_flag=True, metric='zoltar', out_file='outs-cifar/zoltar.txt')
  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.05, attack_flag=True, metric='wong-ii', out_file='outs-cifar/wong.txt')
  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.05, attack_flag=True, metric='ochiai', out_file='outs-cifar/ochiai.txt')
  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.05, attack_flag=True, metric='tarantula', out_file='outs-cifar/tarantula.txt')
