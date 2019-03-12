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

from utils import *
from sbfl import *

img_rows, img_cols, img_channels = 28, 28, 1
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
x_test = x_test.astype('float32')
x_test /= 255

#print ('x data shape:', x.shape)

model=load_model('saved_models/mnist_complicated.h5')

#for idx in range(len(x_test)-1, -1, -1):
for idx in range(0, len(x_test)):
  #idx=np.random.randint(0,len(x_test))
  x=np.array([x_test[idx]])
  y=np.argsort(model.predict(x))[0][-1:]
  
  adv_xs=[]
  adv_ys=[]
  num_advs=0
  
  for c in range(0, 1000):
    sp=x.shape
    sigma=1. #0.75
    noise=np.random.normal(size=sp, scale=sigma)
    adv_x=x+noise
    adv_x=np.clip(adv_x, a_min=0., a_max=1.)
    adv_y=np.argsort(model.predict(adv_x))[0][-1:]
    if adv_y[0]!=y[0]:
      num_advs+=1
    adv_xs.append(adv_x[0])
    adv_ys.append(adv_y[0])
    
  print (num_advs)

  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.10, attack_flag=True, metric='zoltar', out_file='outs/zoltar.txt')
  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.10, attack_flag=True, metric='zoltar', lex_flag=True, out_file='outs/lex-zoltar.txt')
  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.10, attack_flag=True, metric='wong-ii', out_file='outs/wong.txt')
  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.10, attack_flag=True, metric='wong-ii', out_file='outs/lex-wong.txt')
  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.10, attack_flag=True, metric='ochiai', out_file='outs/ochiai.txt')
  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.10, attack_flag=True, metric='ochiai', lex_flag=True, out_file='outs/lex-ochiai.txt')
  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.10, attack_flag=True, metric='tarantula', out_file='outs/tarantula.txt')
  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.10, attack_flag=True, metric='tarantula', out_file='outs/lex-tarantula.txt')
  #raise Exception('stop')
