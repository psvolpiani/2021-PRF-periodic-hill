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

qi=ui
fig, ax1 = plt.subplots(figsize=(10,3))
palette = plt.cm.jet
palette.set_bad ('w',1.0)
A = np.ma.array ( qi, mask=np.isnan(qi))
ax1.pcolor(xi, yi, A, cmap='RdBu', vmin=-0.4, vmax=1.2)
plt.contour(xi,yi,qi,15,linewidths=0.5,colors='k', levels=np.linspace(-0.4, 1.2, 17))
cntr1 = ax1.contourf(xi, yi, qi, 15, cmap="RdBu", levels=np.linspace(-0.4, 1.2, 17))
fig.colorbar(cntr1, ax=ax1)
ax1.set_xlabel( r"$x/h$",fontsize=17)
ax1.set_ylabel( r"$y/h$",fontsize=17)
fig.subplots_adjust(bottom=0.18)
#fig.savefig("fig_dns_19000_u.png", dpi=600)
plt.show()



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

qi=ui
fig, ax1 = plt.subplots(figsize=(10,3))
palette = plt.cm.jet
palette.set_bad ('w',1.0)
A = np.ma.array ( qi, mask=np.isnan(qi))
ax1.pcolor(xi, yi, A, cmap='RdBu', vmin=-0.4, vmax=1.2)
plt.contour(xi,yi,qi,15,linewidths=0.5,colors='k', levels=np.linspace(-0.4, 1.2, 17))
cntr1 = ax1.contourf(xi, yi, qi, 15, cmap="RdBu", levels=np.linspace(-0.4, 1.2, 17))
fig.colorbar(cntr1, ax=ax1)
ax1.set_xlabel( r"$x/h$",fontsize=17)
ax1.set_ylabel( r"$y/h$",fontsize=17)
fig.subplots_adjust(bottom=0.18)
#fig.savefig("fig_rans_19000_u.png", dpi=600)
plt.show()



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

qi=ui
fig, ax1 = plt.subplots(figsize=(10,3))
palette = plt.cm.jet
palette.set_bad ('w',1.0)
A = np.ma.array ( qi, mask=np.isnan(qi))
ax1.pcolor(xi, yi, A, cmap='RdBu', vmin=-0.4, vmax=1.2)
plt.contour(xi,yi,qi,15,linewidths=0.5,colors='k', levels=np.linspace(-0.4, 1.2, 17))
cntr1 = ax1.contourf(xi, yi, qi, 15, cmap="RdBu", levels=np.linspace(-0.4, 1.2, 17))
fig.colorbar(cntr1, ax=ax1)
ax1.set_xlabel( r"$x/h$",fontsize=17)
ax1.set_ylabel( r"$y/h$",fontsize=17)
fig.subplots_adjust(bottom=0.18)
#fig.savefig("fig_nnau_19000_u.png", dpi=600)
plt.show()
