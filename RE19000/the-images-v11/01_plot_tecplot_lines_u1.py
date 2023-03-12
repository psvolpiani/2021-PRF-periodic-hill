#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys
import matplotlib.mathtext as mathtext
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import operator
import pandas as pd

from scipy.interpolate import griddata
from scipy import interpolate
from math import *
from numpy import *
from matplotlib import *
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('lines', linewidth=1.5)
rc('font', size=16)
plt.rc('legend',**{'fontsize':14})
plt.rc({'axes.labelsize': 20})


# -----------------------------
# Define periodic hill function
# -----------------------------

def hill(x):

  h=28.
  L=(144.+2.*54.)/h
  if (x>=0.      and x<9./h  ):
    y = (min( 28., 2.800000000000E+01 + 0.000000000000E+00*(x*28.) + 6.775070969851E-03*(x*28.)**2 - 2.124527775800E-03*(x*28.)**3 ))/28.
  elif (x>=9./h  and x<14./h ):
    y = (2.507355893131E+01 + 9.754803562315E-01*(x*28.) - 1.016116352781E-01*(x*28.)**2 + 1.889794677828E-03*(x*28.)**3)/28.
  elif (x>=14./h and x<20./h ):
    y = (2.579601052357E+01 + 8.206693007457E-01*(x*28.) - 9.055370274339E-02*(x*28.)**2 + 1.626510569859E-03*(x*28.)**3)/28.
  elif (x>=20./h and x<30./h ):
    y = (4.046435022819E+01 - 1.379581654948E+00*(x*28.) + 1.945884504128E-02*(x*28.)**2 - 2.070318932190E-04*(x*28.)**3)/28.
  elif (x>=30./h and x<40./h ):
    y = (1.792461334664E+01 + 8.743920332081E-01*(x*28.) - 5.567361123058E-02*(x*28.)**2 + 6.277731764683E-04*(x*28.)**3)/28.
  elif (x>=40./h and x<54./h ):
    y = (max(0., 5.639011190988E+01 - 2.010520359035E+00*(x*28.) + 1.644919857549E-02*(x*28.)**2 + 2.674976141766E-05*(x*28.)**3))/28.
    
  elif (x<=L       and x>L-9./h  ):
    y = (min( 28., 2.800000000000E+01 - 0.000000000000E+00*((x-L)*28.) + 6.775070969851E-03*((x-L)*28.)**2 + 2.124527775800E-03*((x-L)*28.)**3 ))/28.
  elif (x<=L-9./h  and x>L-14./h  ):
    y = (2.507355893131E+01 - 9.754803562315E-01*((x-L)*28.) - 1.016116352781E-01*((x-L)*28.)**2 - 1.889794677828E-03*((x-L)*28.)**3)/28.
  elif (x<=L-14./h and x>L-20./h  ):
    y = (2.579601052357E+01 - 8.206693007457E-01*((x-L)*28.) - 9.055370274339E-02*((x-L)*28.)**2 - 1.626510569859E-03*((x-L)*28.)**3)/28.
  elif (x<=L-20./h and x>L-30./h  ):
    y = (4.046435022819E+01 + 1.379581654948E+00*((x-L)*28.) + 1.945884504128E-02*((x-L)*28.)**2 + 2.070318932190E-04*((x-L)*28.)**3)/28.
  elif (x<=L-30./h and x>L-40./h  ):
    y = (1.792461334664E+01 - 8.743920332081E-01*((x-L)*28.) - 5.567361123058E-02*((x-L)*28.)**2 - 6.277731764683E-04*((x-L)*28.)**3)/28.
  elif (x<=L-40./h and x>L-54./h  ):
    y = (max(0., 5.639011190988E+01 + 2.010520359035E+00*((x-L)*28.) + 1.644919857549E-02*((x-L)*28.)**2 - 2.674976141766E-05*((x-L)*28.)**3))/28.
  else: y=0.
  
  return y
  
  
  

# ----------------------
# Read tecplot file: DNS
# ----------------------

file = '../da/refSolution/tcpDNS-Re-19000-nutinf-3-mesh-2.dat'


header = 3
nnodes = 30621
nFeatures = 0
nForces = 0
nvar = 9 + nFeatures + nForces
n = 0
data=[]
with open(file) as f:

    lines=f.readlines()

    for line in lines:

        if (n>header-1 and n<nnodes+header):
            myarray = np.fromstring(line, dtype=float, sep=' ')
            data = np.concatenate([data , myarray])
        n=n+1

data = numpy.reshape(data, (nnodes,nvar))
x = data[:,0]
y = data[:,1]
u = data[:,2]
v = data[:,3]
uu = data[:,4]
vv = data[:,5]
uv = data[:,6]
fx_dns = data[:,7]
fy_dns = data[:,8]

# -----------------------------------------------------------------------------------------
# Create a contour plot of irregularly spaced data coordinates via interpolation on a grid.
# -----------------------------------------------------------------------------------------

# Create grid values first.
xi = np.linspace(min(x), max(x), 361)
yi = np.linspace(min(y), max(y), 256)

# Perform linear interpolation of the data (x,y) on a grid defined by (xi,yi)
ui = griddata((x, y), u, (xi[None,:], yi[:,None]), method='linear')
vi = griddata((x, y), v, (xi[None,:], yi[:,None]), method='linear')
qi = griddata((x, y), fx_dns, (xi[None,:], yi[:,None]), method='linear')

for i in range (0,len(xi)):
  for j in range (0,len(yi)):
    if ( yi[j] < hill(xi[i])): ui[j,i] = np.nan
    if ( yi[j] < hill(xi[i])): vi[j,i] = np.nan
    if ( yi[j] < hill(xi[i])): qi[j,i] = np.nan

# -----------------------------------------------------------------------------------------
# Extract profiles from interpolated field.
# -----------------------------------------------------------------------------------------

x0 = []; x1 = []; x2 = []; x3 = []; x4 = []; x5 = []; x6 = []; x7 = []; x8 = [];
y0 = []; y1 = []; y2 = []; y3 = []; y4 = []; y5 = []; y6 = []; y7 = []; y8 = [];

station=0.05
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y0.append(yi[j])
    x0.append(ui[j,I]+station)
    
station=1.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y1.append(yi[j])
    x1.append(ui[j,I]+station)

station=2.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y2.append(yi[j])
    x2.append(ui[j,I]+station)

station=3.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y3.append(yi[j])
    x3.append(ui[j,I]+station)

station=4.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y4.append(yi[j])
    x4.append(ui[j,I]+station)

station=5.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y5.append(yi[j])
    x5.append(ui[j,I]+station)
        
station=6.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y6.append(yi[j])
    x6.append(ui[j,I]+station)

station=7.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y7.append(yi[j])
    x7.append(ui[j,I]+station)

station=8.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y8.append(yi[j])
    x8.append(ui[j,I]+station)

xhill = np.linspace(0., 9., 500)
yhill = np.zeros(len(xhill))
for i in range (len(xhill)): yhill[i] = hill(xhill[i])

fig, ax2 = plt.subplots(figsize=(10,3))
ax2.plot(xhill,yhill,linewidth=1.5,color='gray')
ax2.plot((0,0), (1, 3.035 ),linewidth=1.5,color='gray')
ax2.plot((9,9), (1, 3.035 ),linewidth=1.5,color='gray')
ax2.plot((0,9), (3.035,3.035),linewidth=1.5,color='gray')
ax2.plot(x0,y0,color='k')
ax2.plot(x1,y1,color='k')
ax2.plot(x2,y2,color='k')
ax2.plot(x3,y3,color='k')
ax2.plot(x4,y4,color='k')
ax2.plot(x5,y5,color='k')
ax2.plot(x6,y6,color='k')
ax2.plot(x7,y7,color='k')
ax2.plot(x8,y8,color='k')
ax2.set_xlim([-1, 10])
ax2.set_ylim([-0.05, 3.5])
ax2.xaxis.set_ticks(np.linspace(-1, 10, 12))
ax2.yaxis.set_ticks(np.linspace(0, 3, 4))
ax2.set_xlabel( r"$u/u_b+x/h$",fontsize=18)
ax2.set_ylabel( r"$y/h$",fontsize=18)
fig.subplots_adjust(right=0.90)
fig.subplots_adjust(bottom=0.20)
#fig.savefig("fig_dns_profiles_u.png", dpi=600)
#plt.show()




# ----------------------------
# Read tecplot file: Base RANS
# ----------------------------

file = '../baseRans/tcpSolution-Re-19000-nutinf-3-mesh-2.dat'

header = 3
nnodes = 30621
nvar = 6
n = 0
data=[]
with open(file) as f:

    lines=f.readlines()

    for line in lines:

        if (n>header-1 and n<nnodes+header):
            myarray = np.fromstring(line, dtype=float, sep=' ')
            data = np.concatenate([data , myarray])
        n=n+1

data = numpy.reshape(data, (nnodes,nvar))
x = data[:,0]
y = data[:,1]
u = data[:,2]
v = data[:,3]
p = data[:,4]
n = data[:,5]

# -----------------------------------------------------------------------------------------
# Create a contour plot of irregularly spaced data coordinates via interpolation on a grid.
# -----------------------------------------------------------------------------------------

# Create grid values first.
xi = np.linspace(min(x), max(x), 361)
yi = np.linspace(min(y), max(y), 256)

# Perform linear interpolation of the data (x,y) on a grid defined by (xi,yi)
ui = griddata((x, y), u, (xi[None,:], yi[:,None]), method='linear')

for i in range (0,len(xi)):
  for j in range (0,len(yi)):
    if ( yi[j] < hill(xi[i])): ui[j,i] = np.nan

# -----------------------------------------------------------------------------------------
# Extract profiles from interpolated field.
# -----------------------------------------------------------------------------------------

x0 = []; x1 = []; x2 = []; x3 = []; x4 = []; x5 = []; x6 = []; x7 = []; x8 = [];
y0 = []; y1 = []; y2 = []; y3 = []; y4 = []; y5 = []; y6 = []; y7 = []; y8 = [];

station=0.05
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y0.append(yi[j])
    x0.append(ui[j,I]+station)
    
station=1.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y1.append(yi[j])
    x1.append(ui[j,I]+station)

station=2.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y2.append(yi[j])
    x2.append(ui[j,I]+station)

station=3.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y3.append(yi[j])
    x3.append(ui[j,I]+station)

station=4.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y4.append(yi[j])
    x4.append(ui[j,I]+station)

station=5.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y5.append(yi[j])
    x5.append(ui[j,I]+station)
        
station=6.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y6.append(yi[j])
    x6.append(ui[j,I]+station)

station=7.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y7.append(yi[j])
    x7.append(ui[j,I]+station)

station=8.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y8.append(yi[j])
    x8.append(ui[j,I]+station)


xhill = np.linspace(0., 9., 500)
yhill = np.zeros(len(xhill))
for i in range (len(xhill)): yhill[i] = hill(xhill[i])

ax2.plot(x0,y0,color='C0',linestyle=':')
ax2.plot(x1,y1,color='C0',linestyle=':')
ax2.plot(x2,y2,color='C0',linestyle=':')
ax2.plot(x3,y3,color='C0',linestyle=':')
ax2.plot(x4,y4,color='C0',linestyle=':')
ax2.plot(x5,y5,color='C0',linestyle=':')
ax2.plot(x6,y6,color='C0',linestyle=':')
ax2.plot(x7,y7,color='C0',linestyle=':')
ax2.plot(x8,y8,color='C0',linestyle=':')
##fig.savefig("fig_dns_profiles_u.png", dpi=600)
#plt.show()



# --------------------------
# Read tecplot file: NN RANS
# --------------------------

file = '../newRans-Sc1-au-q10-v11/tcpNN-Re-19000-nutinf-3-mesh-2.dat'

header = 3
nnodes = 30621
nFeatures = 0
nForces = 6
nvar = 6 + nFeatures + nForces
n = 0
data=[]
with open(file) as f:

    lines=f.readlines()

    for line in lines:

        if (n>header-1 and n<nnodes+header):
            myarray = np.fromstring(line, dtype=float, sep=' ')
            data = np.concatenate([data , myarray])
        n=n+1

data = numpy.reshape(data, (nnodes,nvar))
x = data[:,0]
y = data[:,1]
u = data[:,2]
v = data[:,3]
p = data[:,4]
n = data[:,5]
fx_DA = data[:,6]
fy_DA = data[:,7]
fx_SA = data[:,8]
fy_SA = data[:,9]
fx_sum = data[:,10]
fy_sum = data[:,11]

# -----------------------------------------------------------------------------------------
# Create a contour plot of irregularly spaced data coordinates via interpolation on a grid.
# -----------------------------------------------------------------------------------------

# Create grid values first.
xi = np.linspace(min(x), max(x), 361)
yi = np.linspace(min(y), max(y), 256)

# Perform linear interpolation of the data (x,y) on a grid defined by (xi,yi)
ui = griddata((x, y), u, (xi[None,:], yi[:,None]), method='linear')
fx_DAi = griddata((x, y), fx_DA, (xi[None,:], yi[:,None]), method='linear')
fx_SAi = griddata((x, y), fx_SA, (xi[None,:], yi[:,None]), method='linear')
fx_i = griddata((x, y), fx_sum, (xi[None,:], yi[:,None]), method='linear')

for i in range (0,len(xi)):
  for j in range (0,len(yi)):
    if ( yi[j] < hill(xi[i])): ui[j,i] = np.nan
    if ( yi[j] < hill(xi[i])): fx_DAi[j,i] = np.nan
    if ( yi[j] < hill(xi[i])): fx_SAi[j,i] = np.nan
    if ( yi[j] < hill(xi[i])): fx_i[j,i] = np.nan

# -----------------------------------------------------------------------------------------
# Extract profiles from interpolated field.
# -----------------------------------------------------------------------------------------

x0 = []; x1 = []; x2 = []; x3 = []; x4 = []; x5 = []; x6 = []; x7 = []; x8 = [];
y0 = []; y1 = []; y2 = []; y3 = []; y4 = []; y5 = []; y6 = []; y7 = []; y8 = [];

station=0.05
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y0.append(yi[j])
    x0.append(ui[j,I]+station)
    
station=1.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y1.append(yi[j])
    x1.append(ui[j,I]+station)

station=2.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y2.append(yi[j])
    x2.append(ui[j,I]+station)

station=3.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y3.append(yi[j])
    x3.append(ui[j,I]+station)

station=4.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y4.append(yi[j])
    x4.append(ui[j,I]+station)

station=5.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y5.append(yi[j])
    x5.append(ui[j,I]+station)
        
station=6.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y6.append(yi[j])
    x6.append(ui[j,I]+station)

station=7.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y7.append(yi[j])
    x7.append(ui[j,I]+station)

station=8.0
min_index, min_value = min(enumerate(abs(xi[:]-station)), key=operator.itemgetter(1)); I = min_index;
for j in range (0,len(yi)):
  if ( yi[j] >= hill(xi[I])):
    y8.append(yi[j])
    x8.append(ui[j,I]+station)


xhill = np.linspace(0., 9., 500)
yhill = np.zeros(len(xhill))
for i in range (len(xhill)): yhill[i] = hill(xhill[i])

ax2.plot(x0,y0,color='C3',linestyle='-.')
ax2.plot(x1,y1,color='C3',linestyle='-.')
ax2.plot(x2,y2,color='C3',linestyle='-.')
ax2.plot(x3,y3,color='C3',linestyle='-.')
ax2.plot(x4,y4,color='C3',linestyle='-.')
ax2.plot(x5,y5,color='C3',linestyle='-.')
ax2.plot(x6,y6,color='C3',linestyle='-.')
ax2.plot(x7,y7,color='C3',linestyle='-.')
ax2.plot(x8,y8,color='C3',linestyle='-.')
fig.savefig("fig_profiles_19000_u_1nn.pdf", dpi=600)
plt.show()
