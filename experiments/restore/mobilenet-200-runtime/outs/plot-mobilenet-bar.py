import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.ticker import FuncFormatter


params = {'legend.fontsize': 10,
          'font.size':15 }#,
          #'legend.handlelength': 0.01}
plt.rcParams.update(params)

Tot=1 #224.*224.


def read_file(fname):
  c0=[]
  c1=[]
  c2=[]
  c3=[]
  c4=[]
  c5=[]
  with open(fname, 'r') as csvfile:
    res=csv.reader(csvfile, delimiter=' ')
    for row in res:
      c0.append(float(row[0]))
      c1.append(float(row[1]))
      if fname=='shap.txt': break
      c2.append(float(row[2]))
      c3.append(float(row[3]))
      c4.append(float(row[4]))
      c5.append(float(row[5]))
      #if len(c0)==1000: break
  return (np.array(c0), np.array(c1), np.array(c2), np.array(c3), np.array(c4), np.array(c5))

res_zoltar=read_file('zoltar.txt')
res_tarantula=read_file('tarantula.txt')
res_ochiai=read_file('ochiai.txt')
res_wong=read_file('wong-ii.txt')
res_shap=read_file('shap.txt')

print (res_zoltar)
zoltar_t1=np.mean(res_zoltar[2])
zoltar_t2=np.mean(res_zoltar[3])
zoltar_t3=np.mean(res_zoltar[4])
zoltar_t4=np.mean(res_zoltar[5])

tarantula_t1=np.mean(res_tarantula[2])
tarantula_t2=np.mean(res_tarantula[3])
tarantula_t3=np.mean(res_tarantula[4])
tarantula_t4=np.mean(res_tarantula[5])

ochiai_t1=np.mean(res_ochiai[2])
ochiai_t2=np.mean(res_ochiai[3])
ochiai_t3=np.mean(res_ochiai[4])
ochiai_t4=np.mean(res_ochiai[5])

wong_t1=np.mean(res_wong[2])
wong_t2=np.mean(res_wong[3])
wong_t3=np.mean(res_wong[4])
wong_t4=np.mean(res_wong[5])

shap_t1=np.mean(res_shap[1])
print (shap_t1)

A=(zoltar_t1, zoltar_t3, 0)
B=(ochiai_t2, ochiai_t4)
C=(zoltar_t2, zoltar_t4)
D=(tarantula_t2, tarantula_t4, 0)
D2=(0, 0, shap_t1)
E=(wong_t2, wong_t4)

ind = [0, 1]
width = 0.5

n_groups=3
fig, ax = plt.subplots()
#ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
index = np.arange(n_groups)
print (index)
bar_width = 0.35
opacity = 0.5

rects1 = plt.bar(index, A, bar_width,
alpha=opacity,
#color='b',
ecolor='grey', capsize=10)

rects2 = plt.bar(index, D, bar_width,
alpha=opacity,
#color='g',
ecolor='grey', capsize=10,
bottom=A)

rects3 = plt.bar(index, D2, bar_width,
alpha=opacity,
#color='g',
ecolor='grey', capsize=10,
bottom=A)



##plt.xlabel('Person')
plt.ylabel('averaged runtime in $seconds$')
##plt.title('Scores by person')
plt.xticks(index + 0, ('T(x)=2000', 'T(x)=200', 'SHAP'))
#plt.legend()
#
#
#
##ax.set_xlabel('size of an explanation')
##ax.set_ylabel('accumulated explanations')
ax.grid(color='grey', linestyle='--', linewidth=0.25, alpha=0.25)
##
##plt.legend()
#plt.legend((rects1[0]), ('tests generation'), loc = 2, frameon = 'false')
###plt.legend(('Best', 'Zoltar', 'Trantula', 'Ochiai', 'Wong II'),
###           loc='upper right')
##
###plt.show()
plt.savefig("restore-mobilenet-bar.pdf", bbox_inches='tight')
##
##
plt.show()
