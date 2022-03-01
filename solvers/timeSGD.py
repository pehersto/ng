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

from timeit import default_timer as timer
import time

def timeSGD(ops, dnn, solverName, weightInit, sampleData, tSteps, deltat, alpha, Z, batchSize, miniBatchSize, h, nrIter, storeIndx, key):
    start = timer() # for random seed
    # auxiliary variables
    N = Z.shape[0]
    gradNorm = np.zeros((tSteps,2))
    nrInitIter = 10 # number of first time steps to run with large number of iterations
    factorInit = 1 # first nrInitDirect time steps use factorInit*nrIter many iterations
    # store alpha and Z
    alphaStore = np.zeros((alpha.size, storeIndx.size))
    ZStore = np.zeros((Z.flatten().size, storeIndx.size))
    # check if we should store initial condition
    sIndx = jnp.where(storeIndx == 0)
    if sIndx[0].size > 0:
        alphaStore[:, sIndx[0]] = alpha.reshape((-1,1));
        ZStore[:, sIndx[0]] = Z.reshape((-1,1));
    # determine when to print gradient norms
    t1Per = jnp.ceil(tSteps/25.)
    t1PerSN = jnp.ceil(nrIter/50.)
    # update variables
    if(ops.modeName == 'NG'):
        cHat = jnp.zeros((alpha.size+Z.size,))
    elif(ops.modeName == 'F2'):
        cHat = jnp.zeros((alpha.size,))
    else:
        raise Exception('Not implemented')
    ZInit = Z


    # initialize solver
    if(solverName == "adamSGD" or 'adamSGDMin'):
        solverParamStr = ""
        if(miniBatchSize < batchSize):
            raise Exception("Minibatching is not implemented")
        else:
            if(solverName == 'adamSGD'):
                solver = jit(lambda nrIter, h, xSamples, ZInit, alphaZ, t, deltat, cHat: loopGD(ops.objJgrad, nrIter, h, xSamples, ZInit, alphaZ, t, deltat, cHat)[0])
            elif(solverName == 'adamSGDMin'):
                solver = jit(lambda nrIter, h, xSamples, ZInit, alphaZ, t, deltat, cHat: loopGD(ops.objJgrad, nrIter, h, xSamples, ZInit, alphaZ, t, deltat, cHat)[1])
            else:
                raise Exception('Unknown solver name')
    else:
        raise Exception("Unknown solver")

    # time integration
    for tIter in range(1, tSteps+1):
        t = tIter*deltat

        startTime = timer()
        if(tIter > nrInitIter):
            factorInit = 1
        else:
            factorInit = 2
            print("Note: we take ", factorInit, "x", nrIter, " iterations")

        # sample data
        xSamples, key = sampleData(batchSize, key, [], alpha, Z, dnn.knots(Z))
        # initialize weights
        if(weightInit == "reuse"):
            pass
        elif(weightInit == "zero"):
            cHat = jnp.zeros(cHat.shape)
        elif(weightInit == "randn"):
            key, subkey = jax.random.split(key)
            cHat = jax.random.normal(subkey, shape = cHat.shape)
        else:
            raise Exception("Unknown weight init")

        if(ops.modeName == 'NG'):
            alphaZ = jnp.hstack((alpha.reshape((-1,1)), Z)).reshape((-1,))
        elif(ops.modeName == 'F2'):
            alphaZ = alpha.reshape((-1,))
        else:
            raise Exception('Not implemented')

        if(miniBatchSize < batchSize):
            raise Exception('Not implemented')
        else:
            cHat, sgCurr, sgCurrMinNorm = solver(factorInit*nrIter, h, xSamples, ZInit, alphaZ, t, deltat, cHat)

        # update alpha
        if(ops.modeName == 'NG'):
            alpha = alpha + deltat*cHat.reshape((N,-1))[:, 0].reshape((-1,))
            Z = Z + deltat*cHat.reshape((N,-1))[:, 1:]
        elif(ops.modeName == 'F2'):
            alpha = alpha.reshape((-1,)) + deltat*cHat.reshape((N,-1))[:, 0].reshape((-1,))
            Z = ZInit
        else:
            raise Exception('Not implemented')

        # check if we should store
        sIndx = jnp.where(storeIndx == tIter);
        if sIndx[0].size > 0:
            alphaStore[:, sIndx[0]] = alpha.reshape((-1,1));
            ZStore[:, sIndx[0]] = Z.reshape((-1,1));

        gradNorm[tIter-1, 0] = jnp.linalg.norm(sgCurr)
        if(ops.modeName == 'NG'):
            gradNorm[tIter-1, 1] = ops.objJ(ZInit, xSamples, jnp.hstack((alpha.reshape((-1,1)), Z)).reshape((-1,)), t, deltat, cHat)
        elif(ops.modeName == 'F2'):
            gradNorm[tIter-1, 1] = ops.objJ(ZInit, xSamples, alpha, t, deltat, cHat)
        else:
            raise Exception('Not implemented')

        oneStep = timer() - startTime
        if(tIter == 1 or tIter < nrInitIter or tIter % t1Per == 0 or tIter == 1):
            oneStepETA = int((tSteps-tIter)*oneStep)
            oneStepTimeDays = int(oneStepETA//86400)
            oneStepTimeHours = int((oneStepETA - oneStepTimeDays*86400)//3600)
            oneStepTimeMinutes = int((oneStepETA - oneStepTimeDays*86400 - oneStepTimeHours*3600)//60)
            oneStepTimeSeconds = int(oneStepETA - oneStepTimeDays*86400 - oneStepTimeHours*3600 - oneStepTimeMinutes*60)
            print(tIter, '/', tSteps, ': grad norm ', "{:.6e}".format(gradNorm[tIter-1, 0]) + ' (' + "{:.6e}".format(sgCurrMinNorm) + '), obj ', "{:.6e}".format(gradNorm[tIter-1, 1]), ': time per timestep ', "{:.6e}".format(oneStep), '(ETA: {days}-{hours}:{minutes}:{seconds})'.format(days=oneStepTimeDays, hours=oneStepTimeHours, minutes=oneStepTimeMinutes, seconds=oneStepTimeSeconds))

    return alphaStore, ZStore, gradNorm, solverParamStr, [], [], key

def loopGD(getGradFun, nrIter, h, xSamples, ZInit, alphaZ, t, deltat, cHat):
    print('compile time-step solver loopGD')
    m = jnp.zeros((alphaZ.size,))
    v = jnp.zeros((alphaZ.size,))
    val = [h, xSamples, alphaZ, t, deltat, cHat, m, v, ZInit, jnp.zeros((alphaZ.size,)), cHat, 0, jnp.zeros((alphaZ.size,))]

    val[12] = getGradFun(val[8], val[1], val[2], val[3], val[4], val[5])
    val[11] = jnp.linalg.norm(val[12])
    val = jax.lax.fori_loop(0, nrIter, lambda i, v: loopGDbodyHelper(getGradFun, i, v), val)
    return (val[5], val[9], val[11]), (val[10], val[12], val[11])


def loopGDbodyHelper(getGradFun, sIter, val):
    val[9] = getGradFun(val[8], val[1], val[2], val[3], val[4], val[5])
    val[5], val[6], val[7] = adamupdate(val[5], val[0], val[6], val[7], val[9], sIter)

    # store the solution with the lowest gradient norm
    gradNorm = jnp.linalg.norm(val[9])
    val[10], val[11], val[12] = jax.lax.cond(gradNorm <= val[11], lambda val: (val[5], gradNorm, val[9]), lambda val: (val[10], val[11], val[12]), val)

    return val
