import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.ticker import FuncFormatter


params = {'legend.fontsize': 10,
          'font.size':15 }#,
          #'legend.handlelength': 0.01}
plt.rcParams.update(params)

Tot=32*32


def read_file(fname):
  c0=[]
  c1=[]
  with open(fname, 'r') as csvfile:
    res=csv.reader(csvfile, delimiter=' ')
    for row in res:
      c0.append(int(row[0]))
      #c1.append(int(row[1]))
      #if len(c0)==1000: break
  return (np.array(c0), np.array(c1))

res_zoltar=read_file('zoltar.txt')

zoltar=res_zoltar[0][:-1]//1
n=len(zoltar)
random=read_file('random.txt')[0][:n]//1
shap=read_file('shap.txt')[0][:n]//1
tarantula=read_file('tarantula.txt')[0][:n]//1
ochiai=read_file('ochiai.txt')[0][:n]//1
wong=read_file('wong-ii.txt')[0][:n]//1

bests=[]
worsts=[]
for i in range(0, n):
  #bests.append(np.amin([zoltar[i], tarantula[i]]))
  bests.append(np.amin([zoltar[i], tarantula[i],ochiai[i],wong[i]]))
  worsts.append(np.amax([zoltar[i], tarantula[i],ochiai[i],wong[i]]))
  

max_diff=224*224

score_z=np.zeros((max_diff,))
score_t=np.zeros((max_diff,))
score_o=np.zeros((max_diff,))
score_w=np.zeros((max_diff,))
score_bests=np.zeros((max_diff,))
score_random=np.zeros((max_diff,))
score_shap=np.zeros((max_diff,))


for diff in range(0, max_diff):
  for j in range(0, n):
    if bests[j]>=0 and bests[j]<=diff:
      score_bests[diff]+=1
    if random[j]>=0 and random[j]<=diff:
      score_random[diff]+=1
    if zoltar[j]>=0 and zoltar[j]<=diff:
      score_z[diff]+=1
    if tarantula[j]>=0 and tarantula[j]<=diff:
      score_t[diff]+=1
    if ochiai[j]>=0 and ochiai[j]<=diff:
      score_o[diff]+=1
    if wong[j]>=0 and wong[j]<=diff:
      score_w[diff]+=1
    if shap[j]>=0 and shap[j]<=diff:
      score_shap[diff]+=1

score_bests=score_bests/(n*1.0)
score_random=score_random/(n*1.0)
score_shap=score_shap/(n*1.0)
score_z=score_z/(n*1.0)
score_t=score_t/(n*1.0)
score_o=score_o/(n*1.0)
score_w=score_w/(n*1.0)

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
min_=224*224//10-1
#min_=1 
plt.plot(np.array(range(min_, max_diff))/max_diff, score_z[min_:], label='Zoltar', dashes=[2, 2, 10, 2])
plt.plot(np.array(range(min_, max_diff))/max_diff, score_t[min_:], label='Tarantula', ls='-.')
plt.plot(np.array(range(min_, max_diff))/max_diff, score_shap[min_:], label='SHAP', dashes=[2, 10, 2, 2])
plt.plot(np.array(range(min_, max_diff))/max_diff, score_random[min_:], label='Random', dashes=[2, 2, 5, 2])
plt.plot(np.array(range(min_, max_diff))/max_diff, score_w[min_:], label='Wong II', ls='--')
plt.plot(np.array(range(min_, max_diff))/max_diff, score_o[min_:], label='Ochiai', ls=':') 
plt.plot(np.array(range(min_, max_diff))/max_diff, score_bests[min_:], label='DeepCover')

#plt.xlim(100,None)
#plt.xlim(0.1-0.01,1.0+0.025)
ax.set_xticks([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
#plt.ylim(0.2,None)

ax.set_xlabel('partial input size')
ax.set_ylabel('correctly restored decisions')
ax.grid(color='grey', linestyle='--', linewidth=0.25, alpha=0.25)

plt.legend()
#plt.legend(('Best', 'Zoltar', 'Trantula', 'Ochiai', 'Wong II'),
#           loc='upper right')

#plt.show()
plt.savefig("restore-mobilenet.pdf", bbox_inches='tight')
