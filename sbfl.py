from utils import *
import numpy as np
import sys
import os
from datetime import datetime

from sbfl_attack import *

def sbfl(sbfl_element, filter_factor=0.25, metric='zoltar', lex_flag=False, attack_flag=False, out_file=None):
  origin_data=sbfl_element.x
  sp=origin_data.shape
  ef=np.zeros(sp,dtype=float)
  nf=np.zeros(sp,dtype=float)
  ep=np.zeros(sp,dtype=float)
  np_=np.zeros(sp,dtype=float)

  xs=np.array(sbfl_element.xs)

  diffs=xs-origin_data

  diffs=np.abs(diffs)-filter_factor*xs

  F=0
  the_adv=None
  for i in range(0, len(diffs)):
    is_adv=(sbfl_element.y!=sbfl_element.ys[i])
    if is_adv:
      the_adv=sbfl_element.xs[i]
      for index, _ in np.ndenumerate(diffs[i]):
        flag=diffs[i][index]>0
        if flag:
          ef[index]+=1
        else:
          nf[index]+=1
      F+=1
    else:
      for index, _ in np.ndenumerate(diffs[i]):
        flag=diffs[i][index]>0
        if flag:
          ep[index]+=1
        else:
          np_[index]+=1

  ind=None
  spectrum=None
  if metric=='zoltar':
    zoltar=np.zeros(sp, dtype=float)
    for index, x in np.ndenumerate(origin_data):
      aef=ef[index]
      anf=nf[index]
      anp=np_[index]
      aep=ep[index]
      if aef==0:
        zoltar[index]=0
      else:
        k=(10000.0*anf*aep)/aef
        zoltar[index]=(aef*1.0)/(aef+anf+aep+k)
      if lex_flag and aef==F: zoltar[index]=anp+2
    spectrum=zoltar
  elif metric=='wong-ii':
    wong=np.zeros(sp, dtype=float)
    for index, x in np.ndenumerate(origin_data):
      aef=ef[index]
      anf=nf[index]
      anp=np_[index]
      aep=ep[index]
      wong[index]=aef-aep
      if lex_flag and aef==F: wong[index]=anp+2
    spectrum=wong
  elif metric=='ochiai':
    ochiai=np.zeros(sp, dtype=float)
    for index, x in np.ndenumerate(origin_data):
      aef=ef[index]
      anf=nf[index]
      anp=np_[index]
      aep=ep[index]
      ochiai[index]=aef/np.sqrt((aef+anf)*(aef+aep))
      if lex_flag and aef==F: ochiai[index]=anp+2
    spectrum=ochiai
  elif metric=='tarantula':
    tarantula=np.zeros(sp, dtype=float)
    for index, x in np.ndenumerate(origin_data):
      aef=ef[index]
      anf=nf[index]
      anp=np_[index]
      aep=ep[index]
      #tarantula[index]=aef/(aef+anp)
      tarantula[index]=(aef/(aef+anf))/(aef/(aef+anf)+anp/(aep+anp))
      if lex_flag and aef==F: tarantula[index]=anp+2
    spectrum=tarantula
  else:
    raise Exception('The measure is not supported: {0}'.format(metric))

  ind=np.argsort(spectrum, axis=None)

  if lex_flag:
    di='outs/lex-{1}-{0}'.format(str(datetime.now()).replace(' ', '-'), metric)
  else:
    di='outs/{1}-{0}'.format(str(datetime.now()).replace(' ', '-'), metric)
  di=di.replace(':', '-')

  ###
  if attack_flag:
    #sbfl_attack(sbfl_element.model, origin_data, ind, di, out_file=out_file, adv_x=the_adv) 
    sbfl_attack(sbfl_element.model, origin_data, ind, di, out_file=out_file) 
    return 
  ###

  os.system('mkdir -p {0}'.format(di))
  save_an_image(origin_data, 'origin-{0}'.format(sbfl_element.y), di)


  for step in np.arange(10, len(ind), len(ind)//100): 
    zoltar_im=np.ones(sp)
    online_im=sbfl_element.x.copy()
    if sp[2]==1:
      zoltar_im=np.zeros(sp) ## MNIST
    for pos in range(len(ind)-step, len(ind)):
      if sp[2]==1:
        zoltar_im[np.unravel_index(ind[pos], sp)]=1 ## MNIST
      else:
        ipos=np.unravel_index(ind[pos], sp)
        zoltar_im[ipos[0]][ipos[1]][0]=origin_data[ipos[0]][ipos[1]][0]
        online_im[ipos[0]][ipos[1]][0]=255.0
        try:
          zoltar_im[ipos[0]][ipos[1]][1]=origin_data[ipos[0]][ipos[1]][1]
          zoltar_im[ipos[0]][ipos[1]][2]=origin_data[ipos[0]][ipos[1]][2]
          online_im[ipos[0]][ipos[1]][1]=0.
          online_im[ipos[0]][ipos[1]][2]=0.
        except: pass

    save_an_image(zoltar_im, '{1}-{0}'.format(step, metric), di)
    save_an_image(online_im, '{1}-{0}-b'.format(step, metric), di)


#def zoltar(sbfl_element, filter_factor=0.25):
#  origin_data=sbfl_element.x
#  sp=origin_data.shape
#  ef=np.zeros(sp,dtype=float)
#  nf=np.zeros(sp,dtype=float)
#  ep=np.zeros(sp,dtype=float)
#  np_=np.zeros(sp,dtype=float)
#
#  xs=np.array(sbfl_element.xs)
#  print ('xs shape', xs.shape)
#
#  diffs=xs-origin_data
#  print ('xs shape', diffs.shape)
#
#  #diffs=np.abs(diffs)-filter_factor*origin_data
#  diffs=np.abs(diffs)-filter_factor*xs
#
#  the_i=-1
#  for i in range(0, len(diffs)):
#    is_adv=(sbfl_element.y!=sbfl_element.ys[i])
#    if is_adv:
#      the_i=i
#      for index, _ in np.ndenumerate(diffs[i]):
#        #flag=((np.abs(diffs[i][index])-filter_factor*origin_data[index]>0) and (np.abs(diffs[i][index])-filter_factor*xs[i][index]>0))
#        #flag=((np.abs(diffs[i][index])-filter_factor*xs[i][index]>0))
#        flag=diffs[i][index]>0
#        if flag:
#          ef[index]+=1
#        else:
#          nf[index]+=1
#    else:
#      for index, _ in np.ndenumerate(diffs[i]):
#        #flag=((np.abs(diffs[i][index])-filter_factor*origin_data[index]>0) and (np.abs(diffs[i][index])-filter_factor*xs[i][index]>0))
#        #flag=((np.abs(diffs[i][index])-filter_factor*xs[i][index]>0))
#        flag=diffs[i][index]>0
#        if flag:
#          ep[index]+=1
#        else:
#          np_[index]+=1
#
#  print ('to generate zoltar')
#
#  zoltar=np.zeros(sp, dtype=float)
#  for index, x in np.ndenumerate(origin_data):
#    aef=ef[index]
#    anf=nf[index]
#    anp=np_[index]
#    aep=ep[index]
#    if aef==0:
#      zoltar[index]=0
#    else:
#      k=(10000.0*anf*aep)/aef
#      zoltar[index]=(aef*1.0)/(aef+anf+aep+k)
#
#
#  di='zoltar-{0}'.format(str(datetime.now()).replace(' ', '-'))
#  di=di.replace(':', '-')
#  os.system('mkdir -p {0}'.format(di))
#  save_an_image(origin_data, 'origin-{0}'.format(sbfl_element.y), di)
#
#  ind=np.argsort(zoltar, axis=None)
#
#  ###
#  sbfl_attack(sbfl_element.model, origin_data, ind, di) 
#  ###
#  return 
#
#  for step in np.arange(10, len(ind), len(ind)//100): 
#    zoltar_im=np.ones(sp)
#    online_im=sbfl_element.x.copy()
#    if sp[2]==1:
#      zoltar_im=np.zeros(sp) ## MNIST
#    for pos in range(len(ind)-step, len(ind)):
#      if sp[2]==1:
#        zoltar_im[np.unravel_index(ind[pos], sp)]=1 ## MNIST
#      else:
#        ipos=np.unravel_index(ind[pos], sp)
#        zoltar_im[ipos[0]][ipos[1]][0]=origin_data[ipos[0]][ipos[1]][0]
#        online_im[ipos[0]][ipos[1]][0]=255.0
#        try:
#          zoltar_im[ipos[0]][ipos[1]][1]=origin_data[ipos[0]][ipos[1]][1]
#          zoltar_im[ipos[0]][ipos[1]][2]=origin_data[ipos[0]][ipos[1]][2]
#          online_im[ipos[0]][ipos[1]][1]=0.
#          online_im[ipos[0]][ipos[1]][2]=0.
#        except: pass
#
#    save_an_image(zoltar_im, 'zoltar-{0}'.format(step), di)
#    save_an_image(online_im, 'zoltar-{0}-b'.format(step), di)
#
#
