"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import json

def reorder(x_ord, x, y):
  x_ord[:] = x
  x_ord[:],  y[:] = (list(t) for t in zip(*sorted(zip(x,y))))

  return  

jsonfile = open('perf_results_k60.json','r')

json_dict= json.load(jsonfile)

sizes = []
copy_dt = []
copyi1_dt= []
sumi1_dt= []
sumj1_dt=[]
sumk1_dt=[]
avgi_dt=[]
avgj_dt=[]
avgk_dt=[]
lap_dt=[]

copy_dnt = []
copyi1_dnt= []
sumi1_dnt= []
sumj1_dnt=[]
sumk1_dnt=[]
avgi_dnt=[]
avgj_dnt=[]
avgk_dnt=[]
lap_dnt=[]

copy_ft = []
copyi1_ft= []
sumi1_ft= []
sumj1_ft=[]
sumk1_ft=[]
avgi_ft=[]
avgj_ft=[]
avgk_ft=[]
lap_ft=[]

copy_fnt = []
copyi1_fnt= []
sumi1_fnt= []
sumj1_fnt=[]
sumk1_fnt=[]
avgi_fnt=[]
avgj_fnt=[]
avgk_fnt=[]
lap_fnt=[]

greg_sizes = [
8192,
16384,
32768,
65536,
131072,
262144,
524288,
1048576,
2097152,
4194304,
8388608,
16777216,
33554432,
67108864,
134217728,
268435456#,
#536870912,
#1073741824,
#2147483648,
#4294967296 
]

greg_gpu_stream = [
2.58    ,
5.14    ,
10.3    ,
20.5    ,
40.9    ,
80.6    ,
149    ,
272    ,
363    ,
393    ,
461    ,
506    ,
527    ,
543    ,
548    ,
553    #,
#553    ,
#557    ,
#557    ,
#555 
]

greg_model = [
    2.77804909,
    5.542242602,
    11.02933897,
    21.84028239,
    42.82419506,
    82.35590955,
    152.5349863,
    263.2981198,
    402.1332236,
    513.9412596,
    553.6713552,
    556.9801079,
    556.9999993,
    557,
    557,
    557#,
#    557,
#    557,
#    557,
#    557
]

greg_sizes = [x/60. for x in greg_sizes]

for domain in json_dict:
  x = domain['x']
  y = domain['y']
  sizes.append(x*y)
  copy_dt.append( domain['double']['tex']['stencils']['copy'])
  copyi1_dt.append( domain['double']['tex']['stencils']['copyi1'])
  sumi1_dt.append( domain['double']['tex']['stencils']['sumi1'])
  sumj1_dt.append( domain['double']['tex']['stencils']['sumj1'])
  sumk1_dt.append( domain['double']['tex']['stencils']['sumk1'])
  avgi_dt.append( domain['double']['tex']['stencils']['avgi'])
  avgj_dt.append( domain['double']['tex']['stencils']['avgj'])
  avgk_dt.append( domain['double']['tex']['stencils']['avgk'])
  lap_dt.append( domain['double']['tex']['stencils']['lap'])

  copy_dnt.append( domain['double']['no_tex']['stencils']['copy'])
  copyi1_dnt.append( domain['double']['no_tex']['stencils']['copyi1'])
  sumi1_dnt.append( domain['double']['no_tex']['stencils']['sumi1'])
  sumj1_dnt.append( domain['double']['no_tex']['stencils']['sumj1'])
  sumk1_dnt.append( domain['double']['no_tex']['stencils']['sumk1'])
  avgi_dnt.append( domain['double']['no_tex']['stencils']['avgi'])
  avgj_dnt.append( domain['double']['no_tex']['stencils']['avgj'])
  avgk_dnt.append( domain['double']['no_tex']['stencils']['avgk'])
  lap_dnt.append( domain['double']['no_tex']['stencils']['lap'])

  copy_ft.append( domain['float']['tex']['stencils']['copy'])
  copyi1_ft.append( domain['float']['tex']['stencils']['copyi1'])
  sumi1_ft.append( domain['float']['tex']['stencils']['sumi1'])
  sumj1_ft.append( domain['float']['tex']['stencils']['sumj1'])
  sumk1_ft.append( domain['float']['tex']['stencils']['sumk1'])
  avgi_ft.append( domain['float']['tex']['stencils']['avgi'])
  avgj_ft.append( domain['float']['tex']['stencils']['avgj'])
  avgk_ft.append( domain['float']['tex']['stencils']['avgk'])
  lap_ft.append( domain['float']['tex']['stencils']['lap'])

  copy_fnt.append( domain['float']['no_tex']['stencils']['copy'])
  copyi1_fnt.append( domain['float']['no_tex']['stencils']['copyi1'])
  sumi1_fnt.append( domain['float']['no_tex']['stencils']['sumi1'])
  sumj1_fnt.append( domain['float']['no_tex']['stencils']['sumj1'])
  sumk1_fnt.append( domain['float']['no_tex']['stencils']['sumk1'])
  avgi_fnt.append( domain['float']['tex']['stencils']['avgi'])
  avgj_fnt.append( domain['float']['tex']['stencils']['avgj'])
  avgk_fnt.append( domain['float']['tex']['stencils']['avgk'])
  lap_fnt.append( domain['float']['no_tex']['stencils']['lap'])

copy_dt = [x*1024.0*1024.0*1024.0/1e9 for x in copy_dt]
copyi1_dt = [x*1024.0*1024.0*1024.0/1e9 for x in copyi1_dt]
sumi1_dt = [x*1024.0*1024.0*1024.0/1e9 for x in sumi1_dt]
sumj1_dt = [x*1024.0*1024.0*1024.0/1e9 for x in sumj1_dt]
sumk1_dt = [x*1024.0*1024.0*1024.0/1e9 for x in sumk1_dt]
avgi_dt = [x*1024.0*1024.0*1024.0/1e9 for x in avgi_dt]
avgj_dt = [x*1024.0*1024.0*1024.0/1e9 for x in avgj_dt]
avgk_dt = [x*1024.0*1024.0*1024.0/1e9 for x in avgk_dt]
lap_dt = [x*1024.0*1024.0*1024.0/1e9 for x in lap_dt]

copy_dnt = [x*1024.0*1024.0*1024.0/1e9 for x in copy_dnt]
copyi1_dnt = [x*1024.0*1024.0*1024.0/1e9 for x in copyi1_dnt]
sumi1_dnt = [x*1024.0*1024.0*1024.0/1e9 for x in sumi1_dnt]
sumj1_dnt = [x*1024.0*1024.0*1024.0/1e9 for x in sumj1_dnt]
sumk1_dnt = [x*1024.0*1024.0*1024.0/1e9 for x in sumk1_dnt]
avgi_dnt = [x*1024.0*1024.0*1024.0/1e9 for x in avgi_dnt]
avgj_dnt = [x*1024.0*1024.0*1024.0/1e9 for x in avgj_dnt]
avgk_dnt = [x*1024.0*1024.0*1024.0/1e9 for x in avgk_dnt]
lap_dnt = [x*1024.0*1024.0*1024.0/1e9 for x in lap_dnt]

copy_ft = [x*1024.0*1024.0*1024.0/1e9 for x in copy_ft]
copyi1_ft = [x*1024.0*1024.0*1024.0/1e9 for x in copyi1_ft]
sumi1_ft = [x*1024.0*1024.0*1024.0/1e9 for x in sumi1_ft]
sumj1_ft = [x*1024.0*1024.0*1024.0/1e9 for x in sumj1_ft]
sumk1_ft = [x*1024.0*1024.0*1024.0/1e9 for x in sumk1_ft]
avgi_ft = [x*1024.0*1024.0*1024.0/1e9 for x in avgi_ft]
avgj_ft = [x*1024.0*1024.0*1024.0/1e9 for x in avgj_ft]
avgk_ft = [x*1024.0*1024.0*1024.0/1e9 for x in avgk_ft]
lap_ft = [x*1024.0*1024.0*1024.0/1e9 for x in lap_ft]

copy_fnt = [x*1024.0*1024.0*1024.0/1e9 for x in copy_fnt]
copyi1_fnt = [x*1024.0*1024.0*1024.0/1e9 for x in copyi1_fnt]
sumi1_fnt = [x*1024.0*1024.0*1024.0/1e9 for x in sumi1_fnt]
sumj1_fnt = [x*1024.0*1024.0*1024.0/1e9 for x in sumj1_fnt]
sumk1_fnt = [x*1024.0*1024.0*1024.0/1e9 for x in sumk1_fnt]
avgi_fnt = [x*1024.0*1024.0*1024.0/1e9 for x in avgi_fnt]
avgj_fnt = [x*1024.0*1024.0*1024.0/1e9 for x in avgj_fnt]
avgk_fnt = [x*1024.0*1024.0*1024.0/1e9 for x in avgk_fnt]
lap_fnt = [x*1024.0*1024.0*1024.0/1e9 for x in lap_fnt]




sizes_ord=[]

reorder(sizes_ord, sizes, copy_dt)
reorder(sizes_ord, sizes, copyi1_dt)
reorder(sizes_ord, sizes, sumi1_dt)
reorder(sizes_ord, sizes, sumj1_dt)
reorder(sizes_ord, sizes, sumk1_dt)
reorder(sizes_ord, sizes, avgi_dt)
reorder(sizes_ord, sizes, avgj_dt)
reorder(sizes_ord, sizes, avgk_dt)
reorder(sizes_ord, sizes, lap_dt)

reorder(sizes_ord, sizes, copy_dnt)
reorder(sizes_ord, sizes, copyi1_dnt)
reorder(sizes_ord, sizes, sumi1_dnt)
reorder(sizes_ord, sizes, sumj1_dnt)
reorder(sizes_ord, sizes, sumk1_dnt)
reorder(sizes_ord, sizes, avgi_dnt)
reorder(sizes_ord, sizes, avgj_dnt)
reorder(sizes_ord, sizes, avgk_dnt)
reorder(sizes_ord, sizes, lap_dnt)

reorder(sizes_ord, sizes, copy_ft)
reorder(sizes_ord, sizes, copyi1_ft)
reorder(sizes_ord, sizes, sumi1_ft)
reorder(sizes_ord, sizes, sumj1_ft)
reorder(sizes_ord, sizes, sumk1_ft)
reorder(sizes_ord, sizes, avgi_ft)
reorder(sizes_ord, sizes, avgj_ft)
reorder(sizes_ord, sizes, avgk_ft)
reorder(sizes_ord, sizes, lap_ft)

reorder(sizes_ord, sizes, copy_fnt)
reorder(sizes_ord, sizes, copyi1_fnt)
reorder(sizes_ord, sizes, sumi1_fnt)
reorder(sizes_ord, sizes, sumj1_fnt)
reorder(sizes_ord, sizes, sumk1_fnt)
reorder(sizes_ord, sizes, avgi_fnt)
reorder(sizes_ord, sizes, avgj_fnt)
reorder(sizes_ord, sizes, avgk_fnt)
reorder(sizes_ord, sizes, lap_fnt)




fig, ax = plt.subplots()
line1, = ax.plot(sizes_ord, copy_dt, '-o',
                 label='copy_dt')
line1, = ax.plot(sizes_ord, copyi1_dt, '-v', linewidth=2,
                 label='copyi1_dt')
line1, = ax.plot(sizes_ord, sumi1_dt, '-s', linewidth=2,
                 label='sumj1_dt')
line1, = ax.plot(sizes_ord, sumj1_dt, '-^', linewidth=2,
                 label='sumj1_dt')
line1, = ax.plot(sizes_ord, sumk1_dt, '-*', linewidth=2,
                 label='sumk1_dt')
line1, = ax.plot(sizes_ord, avgi_dt, '-h', linewidth=2,
                 label='avgi_dt')
line1, = ax.plot(sizes_ord, avgj_dt, '-d', linewidth=2,
                 label='avgj_dt')
line1, = ax.plot(sizes_ord, avgk_dt, '-p', marker='D', linewidth=2,
                 label='avgk_dt')
line1, = ax.plot(sizes_ord, lap_dt, '-|', linewidth=2,
                 label='lap_dt')
line1, = ax.plot(greg_sizes, greg_gpu_stream, '-*', linewidth=2,
                 label='gpu stream')
line1, = ax.plot(greg_sizes, greg_model, '-*', linewidth=2,
                 label='model')


ax.legend(loc='lower right')
ax.set_xscale("log", nonposy='clip')
ax.set_xlabel("i*j number grid points", fontsize=18)
ax.set_ylabel("mem bandwidth (GB/s)", fontsize=18)


plt.show()
