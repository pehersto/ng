import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax import jit, grad
from functools import partial
# from jax.ops import index, index_add, index_update
from solvers.adam import adamupdate
from datetime import datetime
from scipy import optimize
from scipy import integrate

from timeit import default_timer as timer
import time

kinternalkey = 0
storeTIndx = 0
startTime = 0
resList = []

def timeODE(ops, dnn, schemeName, weightInit, sampleData, tSteps, deltat, alpha, Z, batchSize, miniBatchSize, h, nrIter, storeIndx, sampleForRes, key):
    start = timer() # for random seed
    # auxiliary variables
    N = Z.shape[0]
    gradNorm = np.zeros((tSteps,2))
    # store alpha and Z
    alphaStore = np.zeros((alpha.size, storeIndx.size))
    ZStore = np.zeros((Z.flatten().size, storeIndx.size))
    # check if we should store initial condition
    sIndx = jnp.where(storeIndx == 0)
    if sIndx[0].size > 0:
        alphaStore[:, sIndx[0]] = alpha.reshape((-1,1))
        ZStore[:, sIndx[0]] = Z.reshape((-1,1))
    # determine when to print gradient norms
    t1Per = jnp.ceil(tSteps/25.)
    t1PerSN = jnp.ceil(nrIter/50.)

    maxNrIter = 10000000

    # RHS function
    global kinternalkey
    kinternalkey = key
    def rhsFun(t, alphaZ):
        global storeTIndx
        global kinternalkey
        global resList
        global startTime
        #print(schemeName + ':', t, '/', storeIndx[-1]*deltat)
        if(jnp.isnan(alphaZ).any() or jnp.isinf(alphaZ).any()):
            raise Exception('NaN or Inf in parameters...')
        if(storeTIndx > maxNrIter):
            raise Exception('Maximum number of steps reached', storeTIndx)
        storeTIndx = storeTIndx + 1
        xSamples, kinternalkey = sampleData(batchSize, kinternalkey, [], [], dnn.getZ(alphaZ), [])
        J = ops.rhsJ(Z, xSamples, alphaZ, t)

        if(storeTIndx == 1 or storeTIndx % 50 == 0):
            curResMin = -1
            curResMean = -1
            curResMax = -1
            curResGenMin = -1
            curResGenMean = -1
            curResGenMax = -1
            if(sampleForRes != []):
                curRes = ops.res([], xSamples, alphaZ, t, J)
                curResMin = jnp.min(jnp.square(curRes))
                curResMean = jnp.mean(jnp.square(curRes))
                curResMax = jnp.max(jnp.square(curRes))

                #xSamples = jax.random.uniform(kinternalkey, (batchSize*100, 5), minval = 0., maxval = 6.0)
                xSamples, kinternalkey = sampleForRes(kinternalkey, [], [], dnn.getZ(alphaZ), [])
                curResGen = ops.res([], xSamples, alphaZ, t, J)
                curResGenMin = jnp.min(jnp.square(curResGen))
                curResGenMean = jnp.mean(jnp.square(curResGen))
                curResGenMax = jnp.max(jnp.square(curResGen))
                resList.append([t, curResMin, curResMean, curResMax, curResGenMin, curResGenMean, curResGenMax])

            diffTime = timer() - startTime
            if(t > 0):
                etaTime = diffTime/t*(storeIndx[-1]*deltat - t)
            else:
                etaTime = 0
            oneStepETA = etaTime
            oneStepTimeDays = int(oneStepETA//86400)
            oneStepTimeHours = int((oneStepETA - oneStepTimeDays*86400)//3600)
            oneStepTimeMinutes = int((oneStepETA - oneStepTimeDays*86400 - oneStepTimeHours*3600)//60)
            oneStepTimeSeconds = int(oneStepETA - oneStepTimeDays*86400 - oneStepTimeHours*3600 - oneStepTimeMinutes*60)
            print(schemeName + ':', t, '/', storeIndx[-1]*deltat, storeTIndx, curResMin, curResMean, curResMax, curResGenMin, curResGenMean, curResGenMax, '(ETA: {days}-{hours}:{minutes}:{seconds})'.format(days=oneStepTimeDays, hours=oneStepTimeHours, minutes=oneStepTimeMinutes, seconds=oneStepTimeSeconds))
        return J

    @jit
    def rhsJac(t, alphaZ):
        print('compile rhsJac')
        global kinternalkey
        xSamples, kinternalkey = sampleData(batchSize, kinternalkey, [], [], dnn.getZ(alphaZ), [])
        return jax.jacfwd(lambda az: ops.rhsJ(Z, xSamples, az, t))(alphaZ)

    # update variables
    if(ops.modeName == 'NG'):
        cHat = jnp.hstack((alpha.reshape((-1,1)), Z)).reshape((-1,))
    elif(ops.modeName == 'F2'):
        cHat = alpha.reshape((-1,))
    else:
        raise Exception('Unsupported mode', modeName)
    # ode integration
    global startTime
    startTime = timer()
    t_eval = deltat*storeIndx
    sol = integrate.solve_ivp(rhsFun, [t_eval[0], t_eval[-1]], cHat, method = schemeName, t_eval = t_eval, jac = None, max_step = deltat) #, rtol = 1e-05, atol = 1e-6)

    for i in range(sol.y.shape[1]):
        alphaStore[:, i] = sol.y[:, i].reshape((N,-1))[:, 0]
        if(ops.modeName == 'NG'): # check if we solve in F1 or F2
            ZStore[:, i] = sol.y[:, i].reshape((N,-1))[:, 1:].reshape((-1,))
        elif(ops.modeName == 'F2'):
            ZStore[:, i] = Z.reshape((-1,))
        else:
            raise Exception('Unsupported mode', modeName)

    gradNorm = []
    solverParamStr = ''

    return alphaStore, ZStore, gradNorm, solverParamStr, t_eval, resList, kinternalkey
