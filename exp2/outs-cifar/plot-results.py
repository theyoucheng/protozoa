import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.ticker import FuncFormatter


params = {'legend.fontsize': 12,
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
      c1.append(int(row[1]))
  return (np.array(c0), np.array(c1))

res_zoltar=read_file('zoltar.txt')
res_tarantula=read_file('tarantula.txt')
res_ochiai=read_file('ochiai.txt')
res_wong=read_file('wong.txt')

zoltar=np.array(res_zoltar[0][:-1])/(1.0*np.array(res_zoltar[1][:-1]))
#zoltar=np.array(res_zoltar[0][:1000])/(1.0*np.array(res_zoltar[1][:1000]))
#random=res_zoltar[1][:-1]
n=len(zoltar)
tarantula=np.array(res_tarantula[0][:n])/(1.0*np.array(res_tarantula[1][:n]))
ochiai=np.array(res_ochiai[0][:n])/(1.0*np.array(res_ochiai[1][:n]))
wong=np.array(res_wong[0][:n])/(1.0*np.array(res_wong[1][:n]))

bests=[]
worsts=[]
for i in range(0, n):
  bests.append(np.amin([zoltar[i], tarantula[i],ochiai[i],wong[i]]))
  worsts.append(np.amax([zoltar[i], tarantula[i],ochiai[i],wong[i]]))
  

max_diff=1000 #32*32 #np.amax(worsts)

score_z=np.zeros((max_diff,))
score_t=np.zeros((max_diff,))
score_o=np.zeros((max_diff,))
score_w=np.zeros((max_diff,))
score_bests=np.zeros((max_diff,))

diffs=np.arange(0,max_diff)
diffs=diffs/(max_diff*1.0)

for i in range(0, len(diffs)):
  diff=diffs[i]
  for j in range(0, n):
    if bests[j]>=0 and bests[j]<=diff/1.:
      score_bests[i]+=1
    #if random[j]>=0 and random[j]<=diff/1000.:
    #  score_random[diff]+=1
    if zoltar[j]>=0 and zoltar[j]<=diff/1.:
      score_z[i]+=1
    if tarantula[j]>=0 and tarantula[j]<=diff/1.:
      score_t[i]+=1
    if ochiai[j]>=0 and ochiai[j]<=diff/1.:
      score_o[i]+=1
    if wong[j]>=0 and wong[j]<=diff/1.:
      score_w[i]+=1

score_bests=score_bests/(n*1.0)
#score_random=score_random/(n*1.0)
score_z=score_z/(n*1.0)
score_t=score_t/(n*1.0)
score_o=score_o/(n*1.0)
score_w=score_w/(n*1.0)

#x = np.linspace(0, 2, 100)
#plt.plot(x, x, label='linear')

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.plot(diffs, score_bests, label='Best')
plt.plot(diffs, score_t, label='Tarantula', ls='-.')
plt.plot(diffs, score_o, label='Ochiai', ls=':') 
plt.plot(diffs, score_z, label='Zoltar', dashes=[2, 2, 10, 2])
plt.plot(diffs, score_w, label='Wong II', ls='--')
#plt.plot(range(max_diff), score_random, label='Random', dashes=[2, 10, 2, 2])
#plt.axvline(x=0.22058956)

ax.set_xlabel('speedup factor')
ax.set_ylabel('accumulated explanations')

plt.legend()
#plt.legend(('Best', 'Zoltar', 'Trantula', 'Ochiai', 'Wong II'),
#           loc='upper right')

#plt.show()
plt.savefig("cifar-comparison.pdf", bbox_inches='tight')
