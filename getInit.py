import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit, grad
from solvers import adam

def getInit(mode, prob, dnn, ops, initBatchSize, initNrIter, initH, getInitAZ, initFname, nrReplicates, key):
    '''Train network on initial condition
    Inputs:
        mode            ... mode for training
        prob            ... problem obj
        dnn             ... deep network object
        initBatchSize   ... batch size for SGD
        initNrIter      ... number of iterations
        initH           ... learning rate
        alphaInit       ... starting value alpha
        ZInit           ... starting value Z
        initFname       ... filename for storing/loading
        initDeltat      ... time step for fitting initial condition
    Outputs:
        alpha           ... coefficients of trained network
        Z               ... features of trained network
    '''
    # N = alphaInit.size
    if(mode == "fixedFeatures"):
        ZInit = dnn.initZStatic()
        evale = dnn.evale(prob.plotGrid, ZInit)
        alpha = jnp.linalg.lstsq(evale, prob.u0(prob.plotGrid))[0]
        return alpha, ZInit, key
    elif(mode == 'fixedOrtho'):
        ZInit = dnn.initZStatic()
        evale = dnn.evale(ZInit[:, 1], ZInit)
        alpha = jnp.linalg.solve(evale, prob.u0(ZInit[:, 1]))
        return alpha, ZInit, key
    elif(mode == "fitSGD" or mode == "fitSGD-s" or mode == "fitGD" or mode == "fitGD-s"):
        regParamList = jnp.asarray([0.]) # jnp.hstack((0., jnp.logspace(-5, -2, 8)))
        regError = np.zeros((regParamList.size, nrReplicates))
        azList = []

        mySample = jit(lambda xMat, kkey: prob.sampleDataInit(initBatchSize, kkey, [], [], dnn.getZ(xMat), []))
        myX = jnp.vstack((prob.plotGridInit, prob.plotGrid))
        assert myX.shape[1] == prob.dim and myX.shape[0] == prob.plotGridInit.shape[0] + prob.plotGrid.shape[0]
        for regparam, iter in zip(regParamList, range(regParamList.size)):
            azList.append([])
            for repIter in jnp.arange(nrReplicates):
                alphaInit, ZInit, key = getInitAZ(key)
                N = alphaInit.size
                initAZ = jnp.hstack((alphaInit.reshape((-1,1)), ZInit)).reshape((-1,))

                myobj = lambda dat, xMat: objl2_helper(dnn.ufun, prob.u0, ops.bc, N, regparam, dat, xMat)

                sg = jit(lambda dat, xxMat: grad(myobj, argnums = 1)(dat, xxMat))
                if(mode == "fitSGD" or mode == "fitSGD-s"):
                    curSampleFun = mySample
                elif(mode == "fitGD" or mode == "fitGD-s"):
                    gdsample, key = mySample(initAZ, key)
                    curSampleFun = jit(lambda xMat, kkey: (gdsample, kkey))
                else:
                    raise Exception('Unknown mode')

                az, key = adam.adam(sg, curSampleFun, initH, initNrIter, initAZ, key, printETA = 1, returnBest = 1)

                regError[iter, repIter] = objl2_helper(dnn.ufun, prob.u0, ops.bc, N, 0, myX, az)

                print(regparam, iter, repIter, regError[iter, repIter])
                azList[iter].append(az)

        # select regularization parameter with lowest objective
        bestRegIndxI = np.argmin(np.min(regError, axis = 1))
        bestRegIndxJ = np.argmin(np.min(regError, axis = 0))
        print("Selected:", bestRegIndxI, bestRegIndxJ, regParamList[bestRegIndxI], regError[bestRegIndxI, bestRegIndxJ], regError)
        az = azList[bestRegIndxI][bestRegIndxJ]
        alpha = dnn.getAlpha(az)
        Z = dnn.getZ(az)

        if(mode == "fitSGD" or mode == 'fitGD'):
            pass
        elif(mode == "fitSGD-s" or mode == 'fitGD-s'):
            uInit = dnn.ufunAZ(prob.plotGridInit, alpha, Z)
            uInitAll = dnn.ufunAZ(prob.plotGrid, alpha, Z)
            u0 = prob.u0(prob.plotGridInit)
            u0All = prob.u0(prob.plotGrid)
            jnp.savez(initFname(mode[:-2]), alpha=alpha, Z = Z, initBatchSize=initBatchSize, initH=initH, initNrIter=initNrIter, uInit = [], uInitAll = [], u0Truth = [], u0TruthAll = [], plotGrid = [], plotGridAll = [], regparam = regparam, selRegError = regError[bestRegIndxI, bestRegIndxJ], regError = regError, regParamList = regParamList, knots = [], evalphi = [], evalphiAll = [], OmegaInit = prob.OmegaInit, Omega = prob.Omega)
        else:
            raise Exception("Unknown mode")
    elif(mode == "fitSGD-l" or mode == 'fitGD-l'):
        data = jnp.load(initFname(mode[:-2]))
        alpha = data['alpha']
        Z = data['Z']
    else:
        raise Exception('Unknown init mode')

    return alpha, Z, key

@partial(jit, static_argnums=(0,1,2,3,))
def objl2_helper(ufun, u0fun, bc, N, regparam, x, cHat):
    print('Compile objl2_helper')
    y = u0fun(x).reshape((x.shape[0],))
    return 1./jnp.sqrt(jnp.sum(jnp.square(y)))*jnp.sqrt(jnp.sum(jnp.square(y - ufun(x, cHat)))) + jnp.sum(jnp.square(bc(lambda x: ufun(x, cHat)))) + regparam*jnp.sum(jnp.sum(jnp.square(cHat)))
