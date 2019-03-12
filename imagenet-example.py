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
import sys

img_rows, img_cols, img_channels = 224, 224, 3
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
#x_test = x_test.astype('float32')
#x_test /= 255
#
#idx=np.random.randint(0,len(x_test))
#x=np.array([x_test[idx]])
##print ('x data shape:', x.shape)

model=VGG16()

labels=[555, 920]

xs=[]
print ('To load input data...')
for path, subdirs, files in os.walk('data'):
  for name in files:
    fname=(os.path.join(path, name))
    if fname.endswith('.jpg') or fname.endswith('.png'):
      try:
        image = cv2.imread(fname)
        image = cv2.resize(image, (img_rows, img_cols))
        image=image.astype('float')
        xs.append((image))
        #if len(xs) > 200: break
      except: pass
print ('Total data loaded: ', len(xs))
x_test=np.asarray(xs)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
print ('###', len(x_test))

for idx in range(0, len(x_test)):
  idx=np.random.randint(0,len(x_test))
  x=np.array([x_test[idx]])
  y=np.argsort(model.predict(x))[0][-5:]
  
  
  adv_xs=[]
  adv_ys=[]
  num_advs=0
  
  tot=200*5
  sp=x.shape
  sigma=40.0 #.025*
  noise=np.random.normal(size=(tot, sp[1], sp[2], sp[3]), scale=sigma)
  adv_xs=noise+x
  adv_xs=np.clip(adv_xs, a_min=0., a_max=255.0)
  print ('## to predict')
  adv_labels=model.predict(adv_xs)
  #adv_xs=adv_xs/255.0
  print ('## after the predict')
  for i in range(0, len(adv_labels)):
    adv_y=np.argsort(adv_labels[i])[-5:]
    if len(np.intersect1d(labels, adv_y))==0:
      num_advs+=1
      adv_ys.append(-1)
    else:
      adv_ys.append(0)
  
  if num_advs==0: sys.exit(0)

  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.0, attack_flag=True, metric='zoltar', out_file='outs-imagenet/zoltar.txt')
  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.0, attack_flag=True, metric='wong-ii', out_file='outs-imagenet/wong.txt')
  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.0, attack_flag=True, metric='ochiai', out_file='outs-imagenet/ochiai.txt')
  sbfl(sbfl_elementt(x[0], y[0], adv_xs, adv_ys, model), 0.0, attack_flag=True, metric='tarantula', out_file='outs-imagenet/tarantula.txt')
