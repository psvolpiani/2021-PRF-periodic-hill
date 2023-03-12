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
mesh = input('Insert mesh number:')

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

file = '../../dns/2Dfields_Re'+Re+'.dat'

with open(file, 'rb') as f:

  x    = np.loadtxt(file,skiprows=3, usecols=(0,))
  y    = np.loadtxt(file,skiprows=3, usecols=(1,))
  u    = np.loadtxt(file,skiprows=3, usecols=(2,))
  v    = np.loadtxt(file,skiprows=3, usecols=(3,))

xOutlet = []
yOutlet = []
uOutlet = []
vOutlet = []

for i in range (0,len(x)):
  if (x[i] == 0.):
    xOutlet.append(9.)
    yOutlet.append(y[i])
    uOutlet.append(u[i])
    vOutlet.append(v[i])
    
x = np.concatenate([x , xOutlet])
y = np.concatenate([y , yOutlet])
u = np.concatenate([u , uOutlet])
v = np.concatenate([v , vOutlet])

# ----------------------------------------------------------------------------------------
# Create a contour plot of irregularly spaced data coordinates via interpolation on a grid
# ----------------------------------------------------------------------------------------

# Create grid values first.
xi = np.linspace(min(x), max(x), 1000)
yi = np.linspace(min(y), max(y), 600)

# Perform linear interpolation of the data (x,y) on a grid defined by (xi,yi)
ui = griddata((x, y), u, (xi[None,:], yi[:,None]), method='linear')
vi = griddata((x, y), v, (xi[None,:], yi[:,None]), method='linear')

for i in range (0,len(xi)):
  for j in range (0,len(yi)):
    if ( yi[j] < hill(xi[i])): ui[j,i] = 0. #np.nan
    if ( yi[j] < hill(xi[i])): vi[j,i] = 0. #np.nan


fig, ax1 = plt.subplots(figsize=(10,3))
plt.contour(xi,yi,ui,15,linewidths=0.5,colors='k')
cntr1 = ax1.contourf(xi, yi, ui, 15, cmap="jet")
fig.colorbar(cntr1, ax=ax1)
ax1.set_xlabel( r"$x/h$",fontsize=17)
ax1.set_ylabel( r"$y/h$",fontsize=17)
fig.subplots_adjust(bottom=0.18)
#fig.savefig("fig_dns.png", dpi=600)
plt.show()


# -----------------
# Read tecplot file
# -----------------

file = '../../mesh/tcp-phill-mesh-'+mesh+'.dat'

import pandas as pd
header = 3
nnodes = 30621 # !
nvar = 2
n = 0
data=[]
conn=[]
with open(file) as f:
    
    lines=f.readlines()
    
    for line in lines:
        
        if (n > header-1 and n < nnodes+header):
        
            myarray = np.fromstring(line, dtype=float, sep=' ')
            data = np.concatenate([data , myarray])
            
        n=n+1
        
data = numpy.reshape(data, (nnodes,nvar))
xnew = data[:,0]
ynew = data[:,1]
unew = []
vnew = []

# Interpolate u
f = interpolate.interp2d(xi, yi, ui, kind='linear',fill_value=0.)

for i in range (0,len(xnew)):
  unew = np.concatenate([unew , f(xnew[i], ynew[i])])

# Interpolate v
f = interpolate.interp2d(xi, yi, vi, kind='linear',fill_value=0.)

for i in range (0,len(xnew)):
  vnew = np.concatenate([vnew , f(xnew[i], ynew[i])])
  
for i in range (0,len(xnew)):
  if ( ynew[j] <= hill(xnew[i])):
    unew[i] = 0.
    vnew[i] = 0.

# Perform linear interpolation of the data (x,y) on a grid defined by (xi,yi)
uii = griddata((xnew, ynew), unew, (xi[None,:], yi[:,None]), method='linear')

for i in range (0,len(xi)):
  for j in range (0,len(yi)):
    if ( yi[j] < hill(xi[i])):
      uii[j,i] = np.nan

fig, ax2 = plt.subplots(figsize=(10,3))
plt.contour(xi,yi,uii,15,linewidths=0.5,colors='k')
cntr2 = ax2.contourf(xi, yi, uii, 15, cmap="jet")
fig.colorbar(cntr2, ax=ax2)
ax2.set_xlabel( r"$x/h$",fontsize=17)
ax2.set_ylabel( r"$y/h$",fontsize=17)
fig.subplots_adjust(bottom=0.18)
#fig.savefig("fig_dns_interp.png", dpi=600)
plt.show()


# ---------------------------------------------
# Write tecplot file with interpolated solution
# ---------------------------------------------

#tecplot_file = open("./tecplot_interp_ref_sol_Re"+Re+".dat","w")
#tecplot_file.write('TITLE="solution"\n')
#tecplot_file.write('VARIABLES ="X","Y","U","V"\n')
#tecplot_file.write('ZONE   N=21417,E=42240,F=FEPOINT,ET=TRIANGLE\n')
#tecplot_file.write('21417\n')
#for i in range(len(xnew)):
#    tecplot_file.write( str(xnew[i])+' ' )
#    tecplot_file.write( str(ynew[i])+' ' )
#    tecplot_file.write( str(unew[i])+' ' )
#    tecplot_file.write( str(vnew[i])+' ' )
#    tecplot_file.write( '\n' )
#for i in range(header+nnodes,len(lines)):
#    tecplot_file.write( lines[i] )
#tecplot_file.close()

tecplot_file = open("./tecplot_interp_ref_sol_mesh"+mesh+"_Re"+Re+"-X.dat","w")
tecplot_file.write(str(nnodes)+'\n')
for i in range(len(xnew)):
    tecplot_file.write( str(xnew[i])+' ' )
    tecplot_file.write( '\n' )
tecplot_file.close()

tecplot_file = open("./tecplot_interp_ref_sol_mesh"+mesh+"_Re"+Re+"-Y.dat","w")
tecplot_file.write(str(nnodes)+'\n')
for i in range(len(xnew)):
    tecplot_file.write( str(ynew[i])+' ' )
    tecplot_file.write( '\n' )
tecplot_file.close()

tecplot_file = open("./tecplot_interp_ref_sol_mesh"+mesh+"_Re"+Re+"-U.dat","w")
tecplot_file.write(str(nnodes)+'\n')
for i in range(len(xnew)):
    tecplot_file.write( str(unew[i])+' ' )
    tecplot_file.write( '\n' )
tecplot_file.close()

tecplot_file = open("./tecplot_interp_ref_sol_mesh"+mesh+"_Re"+Re+"-V.dat","w")
tecplot_file.write(str(nnodes)+'\n')
for i in range(len(xnew)):
    tecplot_file.write( str(vnew[i])+' ' )
    tecplot_file.write( '\n' )
tecplot_file.close()
