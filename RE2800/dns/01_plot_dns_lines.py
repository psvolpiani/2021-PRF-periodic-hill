#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys
import matplotlib.mathtext as mathtext
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import operator

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


Re = input('Insert Reynolds number:')

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


# -------------
# Read DNS file
# -------------

file = './2Dfields_Re'+Re+'.dat'

with open(file, 'rb') as f:

  x    = np.loadtxt(file,skiprows=3, usecols=(0,))
  y    = np.loadtxt(file,skiprows=3, usecols=(1,))
  u    = np.loadtxt(file,skiprows=3, usecols=(2,))
  v    = np.loadtxt(file,skiprows=3, usecols=(3,))
  uu   = np.loadtxt(file,skiprows=3, usecols=(6,))
  vv   = np.loadtxt(file,skiprows=3, usecols=(7,))
  ww   = np.loadtxt(file,skiprows=3, usecols=(8,))
  uv   = np.loadtxt(file,skiprows=3, usecols=(9,))

xOutlet = []
yOutlet = []
uOutlet = []
vOutlet = []
uuOutlet = []
vvOutlet = []
wwOutlet = []
uvOutlet = []

for i in range (0,len(x)):
  if (x[i] == 0.):
    xOutlet.append(9.)
    yOutlet.append(y[i])
    uOutlet.append(u[i])
    vOutlet.append(v[i])
    uuOutlet.append(uu[i])
    vvOutlet.append(vv[i])
    wwOutlet.append(ww[i])
    uvOutlet.append(uv[i])
    
x = np.concatenate([x , xOutlet])
y = np.concatenate([y , yOutlet])
u = np.concatenate([u , uOutlet])
v = np.concatenate([v , vOutlet])
uu= np.concatenate([uu, uuOutlet])
vv= np.concatenate([vv, vvOutlet])
ww= np.concatenate([ww, vvOutlet])
uv= np.concatenate([uv, uvOutlet])


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

fig, ax1 = plt.subplots(figsize=(10,3))
plt.contour(xi,yi,ui,15,linewidths=0.5,colors='k')
cntr1 = ax1.contourf(xi, yi, ui, 15, cmap="jet")
fig.colorbar(cntr1, ax=ax1)
ax1.set_xlabel( r"$x/h$",fontsize=17)
ax1.set_ylabel( r"$y/h$",fontsize=17)
fig.subplots_adjust(bottom=0.18)
#fig.savefig("fig_dns.png", dpi=600)
plt.show()


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
plt.plot(xhill,yhill,linewidth=2.,color='k')
plt.plot(x0,y0,color='C0')
plt.plot(x1,y1,color='C0')
plt.plot(x2,y2,color='C0')
plt.plot(x3,y3,color='C0')
plt.plot(x4,y4,color='C0')
plt.plot(x5,y5,color='C0')
plt.plot(x6,y6,color='C0')
plt.plot(x7,y7,color='C0')
plt.plot(x8,y8,color='C0')
ax2.set_xlim([0, 9.2])
ax2.set_xlabel( r"$x/h$",fontsize=17)
ax2.set_ylabel( r"$y/h$",fontsize=17)
fig.subplots_adjust(bottom=0.18)
#fig.savefig("fig_dns_profiles_u.png", dpi=600)
plt.show()

