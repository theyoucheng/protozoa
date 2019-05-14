
import numpy as np
import numpy as np
from keras.models import *
from keras import backend
from keras.preprocessing import image
from keras.models import load_model
#from keras.datasets import mnist
#from keras.datasets import mobilenet
#from keras.applications.vgg16 import VGG16

import sys
sys.path.insert(0, '../../../src/')
from utils import *
from spectra_gen import *
from to_rank import *
from to_restore import *

import shap
import json


img_rows, img_cols, img_channels = 224, 224, 3

xs=[]
imagenet='' ## location to ILSVRC2012_img_val data set, to download http://image-net.org/download
if imagenet=='':
  raise Exception('Please specify the location to ILSVRC2012_img_val data set; to download: http://image-net.org/download')
for path, subdirs, files in os.walk(imagenet):
  for name in files:
    fname=(os.path.join(path, name))
    if fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.JPEG'):
        x=image.load_img(fname, target_size=(img_rows, img_cols))
        x=np.expand_dims(x,axis=0)
        xs.append(x)
        if len(xs)>=10000: break
xs=np.vstack(xs)
xs = xs.reshape(xs.shape[0], img_rows, img_cols, img_channels)
np.random.shuffle(xs)
print ('totally loaded images:', xs.shape)

model = mobilenet.MobileNet(weights='imagenet', include_top=True)

### set up shap
X,y = shap.datasets.imagenet50()
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)
def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [mobilenet.preprocess_input(x.copy())]))
    return K.get_session().run(model.layers[layer].input, feed_dict)


eobj=explain_objectt(model, xs)
eobj.outputs='outs'
eobj.top_classes=1 #5
eobj.adv_ub=1 #.999
eobj.adv_lb=0 #.001
eobj.adv_value= 234 #np.mean(xs, axis=0) #255
#save_an_image(np.mean(xs, axis=0), 'average')
eobj.testgen_factor=0.2
eobj.testgen_size=2000
eobj.testgen_iter=1
eobj.mobilenet=True
#eobj.vgg16=args.vgg16
#eobj.mnist=True
#eobj.cifar10=True
#eobj.inception_v3=args.inception_v3
#eobj.attack=args.attack
#eobj.text_only=args.text_only
#eobj.measure=args.measure
bg_v=255 #eobj.adv_value #255/2

for idx in range(0, len(eobj.inputs)):
  print ('## Input ', idx)
  x=eobj.inputs[idx]
  res=model.predict(sbfl_preprocess(eobj, np.array([x])))
  y=np.argsort(res)[0][-eobj.top_classes:]
  print (y, np.sort(res)[0][-eobj.top_classes:])

  #spectra=spectra_gen(x, adv_value=eobj.adv_value, testgen_factor=eobj.testgen_factor, testgen_size=eobj.testgen_size)
  passing1, failing1=spectra_sym_gen(eobj, x, y[-1:], adv_value=eobj.adv_value, testgen_factor=eobj.testgen_factor, testgen_size=eobj.testgen_size)
  passing2, failing2=spectra_sym_gen(eobj, x, y[-1:], adv_value=eobj.adv_value, testgen_factor=eobj.testgen_factor, testgen_size=eobj.testgen_size)
  #if len(passing)<100: continue
  spectra=[]
  num_advs=len(failing1)
  adv_xs=[]
  adv_ys=[]
  for e in passing1:
    adv_xs.append(e)
    adv_ys.append(0)
  for e in passing2:
    adv_xs.append(e)
    adv_ys.append(0)
  for e in failing1:
    adv_xs.append(e)
    adv_ys.append(-1)
  tot=len(adv_xs)

  adv_part=num_advs*1./tot
  print ('###### adv_percentage:', adv_part, num_advs, tot)

  if adv_part<=eobj.adv_lb:
    print ('###### too few advs')
    continue
  elif adv_part>=eobj.adv_ub:
    print ('###### too many advs')
    continue

  ####
  num_advs=len(failing1)+len(failing2)
  adv_xs2=[]
  adv_ys2=[]
  for e in passing1:
    adv_xs2.append(e)
    adv_ys2.append(0)
  for e in failing1:
    adv_xs2.append(e)
    adv_ys2.append(-1)
  for e in failing2:
    adv_xs2.append(e)
    adv_ys2.append(-1)
  tot=len(adv_xs2)

  adv_part=num_advs*1./tot
  print ('###### double adv_percentage:', adv_part, num_advs, tot)

  if adv_part<=eobj.adv_lb:
    print ('###### too few advs')
    continue
  elif adv_part>=eobj.adv_ub:
    print ('###### too many advs')
    continue


  save_an_image(x, '{0}'.format(idx))
  for measure in ['zoltar', 'wong-ii', 'ochiai', 'tarantula']:
    eobj.measure=measure

    selement=sbfl_elementt(x, 0, adv_xs, adv_ys, model)
    ranking_i=to_rank(selement, eobj.measure)
    eobj.top_classes=5
    partial_x, p_count, p_conf=to_restore(eobj, ranking_i, x, y, bg_v, (img_rows*img_cols)//10, (img_rows*img_cols)//100) #(32*32)//10)
    eobj.top_classes=1
    print ('++', eobj.measure, p_count/(img_rows*img_cols), p_conf)
    save_an_image(partial_x, '{0}-{1}'.format(measure, idx))

    ## testgen_size: 200
    selement200=sbfl_elementt(x, 0, adv_xs2, adv_ys2, model)
    ranking_i200=to_rank(selement200, eobj.measure)
    eobj.top_classes=5
    partial_x200, p_count200, p_conf200=to_restore(eobj, ranking_i200, x, y, bg_v, (img_rows*img_cols)//10, (img_rows*img_cols)//100) #(32*32)//10)
    eobj.top_classes=1
    print ('++', eobj.measure, p_count200/(img_rows*img_cols), p_conf200)
    save_an_image(partial_x200, '{0}-{1}-200'.format(measure, idx))

    ofname='{0}/{1}.txt'.format(eobj.outputs, eobj.measure)
    f = open(ofname,"a") 
    f.write ('{0} {1}\n'.format(p_count, p_count200))           
    f.close()
    #if measure=='zoltar':

  ### shap
  #print ('***', idx)
  #lindex=0
  #e = shap.GradientExplainer(
  #    (model.layers[lindex].input, model.layers[-1].output),
  #    map2layer(X, lindex),
  #    local_smoothing=0 # std dev of smoothing noise
  #)
  #shap_values,indexes = e.shap_values(map2layer(eobj.inputs[idx:idx+1], lindex), ranked_outputs=1)
  #to_explain=np.array(shap_values[0][0])
  #sp=x.shape
  #spectrum=to_explain
  #spectrum_flags=np.zeros(sp, dtype=bool)
  #for iindex, _ in np.ndenumerate(spectrum):
  #  tot=0
  #  for j in range(0, (sp[2])):
  #    if not spectrum_flags[iindex[0]][iindex[1]][j]:
  #      tot+=spectrum[iindex[0]][iindex[1]][j]
  #  for j in range(0, (sp[2])):
  #    if not spectrum_flags[iindex[0]][iindex[1]][j]:
  #      spectrum_flags[iindex[0]][iindex[1]][j]=True
  #      spectrum[iindex[0]][iindex[1]][j]=tot
  #ind=np.argsort(spectrum, axis=None)
  ##partial_x, p_count, p_conf=to_restore(eobj, ind, x, bg_v, (img_rows*img_cols)//100, (img_rows*img_cols)//100) #1) #(32*32)//10)
  #eobj.top_classes=5
  #partial_x, p_count, p_conf=to_restore(eobj, ind, x, y, bg_v, (img_rows*img_cols)//10, (img_rows*img_cols)//100) #(32*32)//10)
  #eobj.top_classes=1
  #print ('++ shap', p_count/(img_rows*img_cols), p_conf)
  #ofname='{0}/{1}.txt'.format(eobj.outputs, 'shap')
  #f = open(ofname,"a") 
  #f.write ('{0} {1}\n'.format(p_count, p_conf))
  #f.close()
  #save_an_image(partial_x, 'shap-{0}'.format(idx))
