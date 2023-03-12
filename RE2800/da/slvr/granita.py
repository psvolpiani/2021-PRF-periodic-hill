#! /usr/bin/env python
# -*- coding:utf-8 -*-

##############################################################################80
# GRANiTA : General Reynolds Avereged Navier sTokes eqs Assimilation algorithm #
##############################################################################80

# system libraies
import sys
import os
import time
from datetime import datetime

# optimisation library
import numpy as np
from scipy.optimize import minimize


# ==============================
# Hardcoded Numerical parameters
# ==============================

# BFGS parameters
verbose       = 2             # verbosity parameter
opttoll       = 1.e-5         # optimization tolerance (def. 1.e-4)
ftoll         = 1.e-7         # optimization tolerance on obj function
gtoll         = 1.e-7         # optimization tolerance on projected gradients
maxlinsear    = 15            # (15) max. iterations for line search algo.
maxiter       = 300           # (~200-500) max. global iterations
applyBounds   = False         # apply constraints or not

# Mesh parameters
Npar          = 2             # number of optimization parameters
Ntri          = 42240         # mesh param
Nver          = 21417         # mesh param
Ndof          = 127072/2      # total number of elements in the mesh
useSlurm4Jobs = False         # use Slurm job scheduling (or sends job manually)

# Paths and file names
dir        = "../results/"
outJ      = dir+"solution-mesh-1"
outDJ     = dir+"gradient-mesh-1"
fileIn     = dir+"field_param"

# Commands
commandsJ = "FreeFem++ solver_J.edp -nw" # -nw: nowindow
commandsDJ= "FreeFem++ solver_DJ.edp"
    

print '--- Start GRANiTA'
if (verbose>0):
    print('Initial parameters')
    print('Npar         =',Npar)
    print('Ndof         =',Ndof)
    print('opttoll      =',opttoll)
    print('ftloo        =',ftoll)
    print('gtoll        =',gtoll)
    print('maxlinsear   =',maxlinsear)
    print('maxiter      =',maxiter)
    print('applyBounds  =',applyBounds)


# initialize by zero or read initialization RANS files
v0      = np.zeros(Ndof*Npar)
filename = fileIn+".txt"
np.savetxt(filename,v0)

# local iterators on getJ and getDJ calls
iterJ      = 1
iterDJ     = 1


###################################################
### Define function(s) for optimisation library ###
###################################################

def getJ(v):
    global iterJ, commandsJ
    if (verbose>9): print "  -- call to getJ:  v = ",v
    
    # write variables as exchange file
    filename = fileIn+".txt"
    np.savetxt(filename,v)
    
    # --- run solver: ---
    os.system(commandsJ+" -iterJ %04d"%iterJ)
    commands = ("cp "+outJ+".txt "+outJ+"-iter%04d.txt"%iterJ)
    os.system(commands)
    print(datetime.now())

    # load newly obtained data from file
    filename = dir+"J.txt"
    J = np.loadtxt(filename)
    if (verbose>1): print "  -- J = ",J
    
    filename = open(dir+"logJ.txt","a")
    filename.write(str(iterJ)+" "+str(J)+"\n")
    filename.close()
    time.sleep(60)
    
    iterJ = iterJ + 1
    # output
    return J

def getdJ(v):
    global iterDJ, commandsDJ
    if (verbose>9): print "  -- call to getdJ: v = ",v
    
    # --- run solver for the gradient (adjoint, etc...) ---
    os.system(commandsDJ)
    commands = ("cp "+outDJ+".txt "+outDJ+"-iter%04d.txt"%iterDJ)
    os.system(commands)

    # load newly generated data from file(s)
    filename   = outDJ+"-iter%04d.txt"%iterDJ
    dJ         = np.loadtxt(filename)
        
    iterDJ = iterDJ + 1
    # output
    return dJ


# ============
# Optimisation
# ============

if (verbose>0): print; print 'Start optimisation:'

res = minimize(getJ, v0, method='L-BFGS-B',jac=getdJ, tol=opttoll, bounds=None,
	    options={'maxls': maxlinsear, 'maxiter': maxiter, 'ftol': ftoll, 'gtol': gtoll, 'disp': (verbose>1)})
v = res.x

# display result
if (verbose>0): print "Optimal parameters: "
if (verbose>9): print v
if (verbose>0): print; print 'End of Optimisation:'

# write variables as exchange file
filename = fileIn+"_opt.txt"
np.savetxt(filename,v)

print "--- End GRANiTA"
