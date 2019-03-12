from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import SaliencyMapMethod

import numpy as np
from utils import *

def sbfl_attack(model, x, ranked_indices, di, adv_x=None, out_file=None):
  sp=x.shape
  y=np.argsort(model.predict(np.array([x])))[0][-1:]

  ## to pick up one adversarial example
  x_1=x.copy()
  wrap = KerasModelWrapper(model)
  sess =  backend.get_session()
  fgsm = FastGradientMethod(wrap, sess=sess)
  ## to detect the min eps
  adv_flag=False
  #for eps in np.arange(0.01, 0.3, 0.01):
  for eps in np.arange(0.003, 0.3, 0.003):
    fgsm_params = {'eps': eps,
                   'clip_min': 0.,
                   'clip_max': 1.}
    if adv_x is None:
      x_1b=fgsm.generate_np(np.array([x_1]), **fgsm_params)
    else:
      x_1b=np.array([adv_x])
    y_1b=np.argsort(model.predict(x_1b))[0][-1:]
    if y_1b[0]!=y[0]:
      ####
      if adv_x is None:
        if sp[2]==1:
          eps+=0.1
        else: eps+=0.3
        fgsm_params = {'eps': eps,
                     'clip_min': 0.,
                     'clip_max': 1.}
        x_1b=fgsm.generate_np(np.array([x_1]), **fgsm_params)
        y_1b=np.argsort(model.predict(x_1b))[0][-1:]
        if y_1b[0]==y[0]: return
      ####
      adv_flag=True
      print ('found adversarial example with eps =', eps)
      c_diffs=(np.count_nonzero(np.array([x])-x_1b))
      ## OK - got the adversarial example
      if sp[2]==1:
        v=np.random.uniform(0,1)
        x_2=x_1.copy()
        adv_p_count=0
        for pos in range(len(ranked_indices)-1, -1, -1):
          index=np.unravel_index(ranked_indices[pos], sp)
          if True: #x_2[index]!=x_1b[0][index]:
            if x_2[index]>0.5:
              x_2[index]=0 #x_1b[0][index]
            else:
              x_2[index]=0 #x_1b[0][index]
            adv_p_count+=1
          y_2b=np.argsort(model.predict(np.array([x_2])))[0][-1:]
          if y_2b!=y[0]: #y_1b:
            #print ('refacting adv with {0} over {1}'.format(adv_p_count, c_diffs))
            break

        ## any benefit from random
        #x_3=x_1.copy()
        ##shuffled_ind=ranked_indices.copy()
        ##np.random.shuffle(shuffled_ind)
        #random_p_count=0
        #for pos in range(len(shuffled_ind)-1, -1, -1):
        #  index=np.unravel_index(shuffled_ind[pos], sp)
        #  if x_3[index]!=x_1b[0][index]:
        #    x_3[index]=x_1b[0][index]
        #    random_p_count+=1
        #  y_3b=np.argsort(model.predict(np.array([x_3])))[0][-1:]
        #  if y_3b==y_1b:
        #    print ('refacting adv (random) with {0} over {1}'.format(random_p_count, c_diffs))
        #    break
        x_3=x_1.copy()
        random_p_count=0
        for pos in range(0, len(ranked_indices)):
          index=np.unravel_index(ranked_indices[pos], sp)
          if True: 
            #x_3[index]=0 #1-x_1[index[0]][index[1]][0] #v1 #x_1b[0][index[0]][index[1]][0]
            if x_3[index]>0.5:
              x_3[index]=0 #x_1b[0][index]
            else:
              x_3[index]=0 #x_1b[0][index]
            random_p_count+=1
            y_3b=np.argsort(model.predict(np.array([x_3])))[0][-1:]
            if y_3b!=y[0]:
              break
      
      else:
        x_3=x_1.copy()
        shuffled_ind=ranked_indices.copy()
        np.random.shuffle(shuffled_ind)
        random_p_count=0
        #for pos in range(len(shuffled_ind)-1, -1, -1):
        #  index=np.unravel_index(shuffled_ind[pos], sp)
        for pos in range(0, len(ranked_indices)):
          index=np.unravel_index(ranked_indices[pos], sp)
          if True: #x_3[index[0]][index[1]][index[2]]!=0: #True: #x_3[index[0]][index[1]][index[2]]!=x_1b[0][index[0]][index[1]][index[2]]:
            x_3[index[0]][index[1]][index[2]]=1 #1-x_1[index[0]][index[1]][0] #v1 #x_1b[0][index[0]][index[1]][0]
            #x_3[index[0]][index[1]][0]=0 #1-x_1[index[0]][index[1]][0] #v1 #x_1b[0][index[0]][index[1]][0]
            #x_3[index[0]][index[1]][1]=0 #1-x_1[index[0]][index[1]][1] #v2 #x_1b[0][index[0]][index[1]][1]
            #x_3[index[0]][index[1]][2]=0 #1-x_1[index[0]][index[1]][2] #v3 #x_1b[0][index[0]][index[1]][2]
            random_p_count+=1
            y_3b=np.argsort(model.predict(np.array([x_3])))[0][-1:]
            #if y_3b==y_1b:
            if y_3b!=y[0]:
              break

        x_2=x_1.copy()
        adv_p_count=0
        for pos in range(len(ranked_indices)-1, -1, -1):
          index=np.unravel_index(ranked_indices[pos], sp)
          #print (index)
          #if x_2[index[0]][index[1]][0]!=x_1b[0][index[0]][index[1]][0] or x_2[index[0]][index[1]][1]!=x_1b[0][index[0]][index[1]][1] or x_2[index[0]][index[1]][2]!=x_1b[0][index[0]][index[1]][2]:
          #  x_2[index[0]][index[1]][0]=x_1b[0][index[0]][index[1]][0]
          #  x_2[index[0]][index[1]][1]=x_1b[0][index[0]][index[1]][1]
          #  x_2[index[0]][index[1]][2]=x_1b[0][index[0]][index[1]][2]
          if True: #x_2[index[0]][index[1]][index[2]]!=0: #True: #x_2[index[0]][index[1]][index[2]]!=x_1b[0][index[0]][index[1]][index[2]]:
            x_2[index[0]][index[1]][index[2]]=1 #1-x_1[index[0]][index[1]][0] #v1 #x_1b[0][index[0]][index[1]][0]
            #x_2[index[0]][index[1]][index[2]]=x_1b[0][index[0]][index[1]][index[2]]
            #x_2[index[0]][index[1]][0]=0 #1-x_1[index[0]][index[1]][0] #v1 #x_1b[0][index[0]][index[1]][0]
            #x_2[index[0]][index[1]][1]=0 #1-x_1[index[0]][index[1]][1] #v2 #x_1b[0][index[0]][index[1]][1]
            #x_2[index[0]][index[1]][2]=0 #1-x_1[index[0]][index[1]][2] #v3 #x_1b[0][index[0]][index[1]][2]
            adv_p_count+=1
            y_2b=np.argsort(model.predict(np.array([x_2])))[0][-1:]
            #if y_2b==y_1b:
            if y_2b!=y[0]:
              #adv_p_count=(np.count_nonzero(np.array([x_2])-x_1b))/3
              #print ('refacting adv with {0} over {1}'.format(adv_p_count, c_diffs))
              #raise Exception('stop point {0}'.format(adv_p_count))
              break

        ### any benefit from random
        #x_3=x_1.copy()
        #shuffled_ind=ranked_indices.copy()
        #np.random.shuffle(shuffled_ind)
        #random_p_count=0
        #for pos in range(len(shuffled_ind)-1, -1, -1):
        #  index=np.unravel_index(shuffled_ind[pos], sp)
        #  #if True: #x_3[index[0]][index[1]][0]!=x_1b[0][index[0]][index[1]][0] or x_3[index[0]][index[1]][1]!=x_1b[0][index[0]][index[1]][1] or x_3[index[0]][index[1]][2]!=x_1b[0][index[0]][index[1]][2]:
        #    #x_3[index[0]][index[1]][0]=x_1b[0][index[0]][index[1]][0]
        #    #x_3[index[0]][index[1]][1]=x_1b[0][index[0]][index[1]][1]
        #    #x_3[index[0]][index[1]][2]=x_1b[0][index[0]][index[1]][2]
        #  if x_3[index[0]][index[1]][index[2]]!=x_1b[0][index[0]][index[1]][index[2]]:
        #    x_3[index[0]][index[1]][0]=x_1b[0][index[0]][index[1]][0]
        #    x_3[index[0]][index[1]][1]=x_1b[0][index[0]][index[1]][1]
        #    x_3[index[0]][index[1]][2]=x_1b[0][index[0]][index[1]][2]
        #    random_p_count+=1
        #    y_3b=np.argsort(model.predict(np.array([x_3])))[0][-1:]
        #    if y_3b==y_1b:
        #      break
      
      break

  if not adv_flag: 
    return

  p_count=0
  im=np.ones(sp)
  for pos in range(len(ranked_indices)-1, -1, -1):
    #if sp[2]==1: ## grey-scale
      index=np.unravel_index(ranked_indices[pos], sp)
      if sp[2]==1:
        im[index]=x[index]
      else:
        im[index[0]][index[1]][0]=x[index[0]][index[1]][0]
        im[index[0]][index[1]][1]=x[index[0]][index[1]][1]
        im[index[0]][index[1]][2]=x[index[0]][index[1]][2]
      p_count=(np.count_nonzero(np.array(im)-x))/sp[2]
      attacked_y=np.argsort(model.predict(np.array([im])))[0][-1:]
      if attacked_y[0]==y[0]:
        #print ('*** found identical example at step {0}***'.format(p_count))
        #save_an_image(im, '0-min-{0}-y{1}'.format(len(ranked_indices)-pos, attacked_y[0]), di)
        if not (out_file is None):
          f = open(out_file,"a") 
        elif sp[2]==1:
          f = open("out-mnist.txt","a") 
        else:
          f = open("out-cifar.txt","a") 
        f.write ('{0} {1} {2} {3} {4}\n'.format(int(adv_p_count), int(random_p_count), c_diffs, p_count, eps))
        f.close()
        return

  

#def sbfl_attack(model, x, ranked_indices, di, adv_x=None, out_file=None):
#  sp=x.shape
#  y=np.argsort(model.predict(np.array([x])))[0][-1:]
#
#  ## to pick up one adversarial example
#  x_1=x.copy()
#  wrap = KerasModelWrapper(model)
#  sess =  backend.get_session()
#  fgsm = FastGradientMethod(wrap, sess=sess)
#  ## to detect the min eps
#  adv_flag=False
#  #for eps in np.arange(0.01, 0.3, 0.01):
#  for eps in np.arange(0.003, 0.3, 0.003):
#    fgsm_params = {'eps': eps,
#                   'clip_min': 0.,
#                   'clip_max': 1.}
#    if adv_x is None:
#      x_1b=fgsm.generate_np(np.array([x_1]), **fgsm_params)
#    else:
#      x_1b=np.array([adv_x])
#    y_1b=np.argsort(model.predict(x_1b))[0][-1:]
#    if y_1b[0]!=y[0]:
#      ####
#      if sp[2]==1:
#        eps+=0.1
#      else: eps+=0.3
#      fgsm_params = {'eps': eps,
#                   'clip_min': 0.,
#                   'clip_max': 1.}
#      x_1b=fgsm.generate_np(np.array([x_1]), **fgsm_params)
#      y_1b=np.argsort(model.predict(x_1b))[0][-1:]
#      if y_1b[0]==y[0]: return
#      ####
#      adv_flag=True
#      print ('found adversarial example with eps =', eps)
#      c_diffs=(np.count_nonzero(np.array([x])-x_1b))
#      ## OK - got the adversarial example
#      if sp[2]==1:
#        x_2=x_1.copy()
#        adv_p_count=0
#        for pos in range(len(ranked_indices)-1, -1, -1):
#          index=np.unravel_index(ranked_indices[pos], sp)
#          if x_2[index]!=x_1b[0][index]:
#            x_2[index]=x_1b[0][index]
#            adv_p_count+=1
#          y_2b=np.argsort(model.predict(np.array([x_2])))[0][-1:]
#          if y_2b==y_1b:
#            print ('refacting adv with {0} over {1}'.format(adv_p_count, c_diffs))
#            break
#
#        ## any benefit from random
#        x_3=x_1.copy()
#        shuffled_ind=ranked_indices.copy()
#        np.random.shuffle(shuffled_ind)
#        random_p_count=0
#        for pos in range(len(shuffled_ind)-1, -1, -1):
#          index=np.unravel_index(shuffled_ind[pos], sp)
#          if x_3[index]!=x_1b[0][index]:
#            x_3[index]=x_1b[0][index]
#            random_p_count+=1
#          y_3b=np.argsort(model.predict(np.array([x_3])))[0][-1:]
#          if y_3b==y_1b:
#            print ('refacting adv (random) with {0} over {1}'.format(random_p_count, c_diffs))
#            break
#      
#      else:
#        x_2=x_1.copy()
#        for pos in range(len(ranked_indices)-1, -1, -1):
#          index=np.unravel_index(ranked_indices[pos], sp)
#          #print (index)
#          x_2[index[0]][index[1]][0]=x_1b[0][index[0]][index[1]][0]
#          x_2[index[0]][index[1]][1]=x_1b[0][index[0]][index[1]][1]
#          x_2[index[0]][index[1]][2]=x_1b[0][index[0]][index[1]][2]
#          y_2b=np.argsort(model.predict(np.array([x_2])))[0][-1:]
#          if y_2b==y_1b:
#            adv_p_count=(np.count_nonzero(np.array([x_2])-x_1b))/3
#            print ('refacting adv with {0} over {1}'.format(adv_p_count, c_diffs))
#            #raise Exception('stop point {0}'.format(adv_p_count))
#            break
#
#        ## any benefit from random
#        x_3=x_1.copy()
#        shuffled_ind=ranked_indices.copy()
#        np.random.shuffle(shuffled_ind)
#        random_p_count=0
#        for pos in range(len(shuffled_ind)-1, -1, -1):
#          index=np.unravel_index(shuffled_ind[pos], sp)
#          #if x_3[index]!=x_1b[0][index]:
#          #  x_3[index]=x_1b[0][index]
#          #  random_p_count+=1
#          x_3[index[0]][index[1]][0]=x_1b[0][index[0]][index[1]][0]
#          x_3[index[0]][index[1]][1]=x_1b[0][index[0]][index[1]][1]
#          x_3[index[0]][index[1]][2]=x_1b[0][index[0]][index[1]][2]
#          y_3b=np.argsort(model.predict(np.array([x_3])))[0][-1:]
#          if y_3b==y_1b:
#            random_p_count=(np.count_nonzero(np.array([x_3])-x_1b))/3
#            print ('refacting adv (random) with {0} over {1}'.format(random_p_count, c_diffs))
#            break
#      
#      break
#
#  if not adv_flag: 
#    return
#
#  p_count=0
#  im=np.ones(sp)
#  for pos in range(len(ranked_indices)-1, -1, -1):
#    #if sp[2]==1: ## grey-scale
#      index=np.unravel_index(ranked_indices[pos], sp)
#      if sp[2]==1:
#        im[index]=x[index]
#      else:
#        im[index[0]][index[1]][0]=x[index[0]][index[1]][0]
#        im[index[0]][index[1]][1]=x[index[0]][index[1]][1]
#        im[index[0]][index[1]][2]=x[index[0]][index[1]][2]
#      p_count=(np.count_nonzero(np.array(im)-x))/sp[2]
#      attacked_y=np.argsort(model.predict(np.array([im])))[0][-1:]
#      if attacked_y[0]==y[0]:
#        #print ('*** found identical example at step {0}***'.format(p_count))
#        #save_an_image(im, '0-min-{0}-y{1}'.format(len(ranked_indices)-pos, attacked_y[0]), di)
#        if not (out_file is None):
#          f = open(out_file,"a") 
#        elif sp[2]==1:
#          f = open("out-mnist.txt","a") 
#        else:
#          f = open("out-cifar.txt","a") 
#        f.write ('{0} {1} {2} {3} {4}\n'.format(int(adv_p_count), int(random_p_count), c_diffs, p_count, eps))
#        f.close()
#        return
#
#  
