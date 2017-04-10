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

jsonfile = open('perf_results.json','r')

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
line1, = ax.plot(sizes_ord, copy_fnt, '-o',
                 label='copy_fnt')
line1, = ax.plot(sizes_ord, copyi1_fnt, '-v', linewidth=2,
                 label='copyi1_fnt')
line1, = ax.plot(sizes_ord, sumi1_fnt, '-s', linewidth=2,
                 label='sumj1_fnt')
line1, = ax.plot(sizes_ord, sumj1_fnt, '-^', linewidth=2,
                 label='sumj1_fnt')
line1, = ax.plot(sizes_ord, sumk1_fnt, '-*', linewidth=2,
                 label='sumk1_fnt')
line1, = ax.plot(sizes_ord, avgi_fnt, '-h', linewidth=2,
                 label='avgi_fnt')
line1, = ax.plot(sizes_ord, avgj_fnt, '-d', linewidth=2,
                 label='avgj_fnt')
line1, = ax.plot(sizes_ord, avgk_fnt, '-p', marker='D', linewidth=2,
                 label='avgk_fnt')
line1, = ax.plot(sizes_ord, lap_fnt, '-|', linewidth=2,
                 label='lap_fnt')

ax.legend(loc='lower right')
ax.set_xscale("log", nonposy='clip')
ax.set_xlabel("i*j number grid points", fontsize=18)
ax.set_ylabel("mem bandwidth (GiB/s)", fontsize=18)


plt.show()
