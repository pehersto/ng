import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from timeit import default_timer as timer
from datetime import datetime
import re

# debug
# from timeit import default_timer as timer
# import data_profiler as profiler

# PyNG specific packages
from ops.OpsKdV import OpsKdV
from ops.OpsAdv import OpsAdv
from ops.OpsParticleTrap import OpsParticleTrap
from Problem import Problem
from getInit import getInit
from solvers.timeODE import timeODE
from solvers.timeSGD import timeSGD
from DNN import DNN
from misc import pyngplot


def testpyNG(modeName, probName, unitName, schemeName, solverName, weightInit, sampleName, N, nrLayers, batchSize, miniBatchSize, h, deltat, nrIter, initSampleName, initBatchSize, initNrIter, initH, zInitMode, nrReplicates, plotInit, plotFinal, writeVideo, onlyInit, writeToFile, postfix):
    start = timer() # for random seed
    print("# probName", probName)
    print("# unitName", unitName)
    print("# schemeName", schemeName)
    print("# solverName", solverName)
    print("# weightInit", weightInit)
    print("# sampleName", sampleName)
    print("# N", N)
    print("# nrLayers", nrLayers)
    print("# batchSize", batchSize)
    print("# miniBatchSize", miniBatchSize)
    print("# h", h)
    print("# deltat", deltat)
    print("# nrIter", nrIter)
    print("# initSampleName", initSampleName)
    print("# initBatchSize", initBatchSize)
    print("# initNrIter", initNrIter)
    print("# initH", initH)
    print("# zInitMode", zInitMode)
    print("# nrReplicates", nrReplicates)
    print("# plotInit", plotInit)
    print("# plotFinal", plotFinal)
    print("# writeVideo", writeVideo)
    print("# onlyInit", onlyInit)
    print("# writeToFile", writeToFile)
    print("# postfix", postfix)

    # random seed with time
    tseed = datetime.now().timestamp() + (timer() - start)
    key = jax.random.PRNGKey(int(tseed*1000))

    # initialize
    prob = Problem(probName, sampleName, N, nrLayers, deltat)
    dnn = DNN(unitName, N, nrLayers, prob.dim, prob.Omega)
    ops = getOps(prob, dnn, schemeName, modeName)

    if(sampleName[:5] == 'gauss' and not (unitName == 'RBF' or unitName == 'RBFp' or unitName == 'RBFNorm' or unitName == 'tanh')):
        print(sampleName, unitName, 'combination not supported')
        return

    # determine when to store
    nrTSteps = int(jnp.round(jnp.divide(prob.Tend, deltat)))
    tIndx = jnp.arange(0, nrTSteps+1, int(max([1, jnp.floor(nrTSteps/75.)])));
    if(tIndx[-1] != nrTSteps): # make sure the final time step is included
        tIndx = jnp.hstack((tIndx, nrTSteps))
    tTimes = tIndx*deltat;

    # check if we have truth solution or if we need to compute/load it
    if(prob.uT != None):
        # truth solution available
        Ubenchmark = lambda x, i: prob.uT(x, jnp.asarray([tTimes[i]])).reshape((-1,))
        FDxGrid = prob.plotGrid
    else:
        UbenchmarkArray, FDxGrid = computeFDSolution(probName, prob.Omega, deltat, prob.Tend, prob.nu, prob.u0, tIndx)
        Ubenchmark = lambda x, i: UbenchmarkArray[:, i]

    # filenames
    initFname = lambda zmode: 'initC/' + probName + '_' + unitName + '_ZI' + zmode + '-' + initSampleName + '-' + "{:.0e}".format(initBatchSize) + '-' + "{:.0e}".format(initNrIter) + '-' + "{:.0e}".format(initH) + '-' + "{:}".format(nrReplicates) + '_N' + "{:}".format(N) + "_nrL" + "{:}".format(nrLayers) + '.npz'

    # init values
    getInitAZ = lambda key: dnn.getInitAZ(key, prob.OmegaInit)
    alphaInit, ZInit, key = getInitAZ(key)

    # initialize
    if((prob.getInitRBF == [] and prob.getInitRBFNorm == [] and prob.getInitRBFp == [] and prob.getInitTanh == []) or zInitMode[:6] == 'fitSGD' or zInitMode[:5] == 'fixed'):
        alphaInit, ZInit, key = getInit(zInitMode, prob, dnn, ops, initBatchSize, initNrIter, initH, getInitAZ, initFname, nrReplicates, key)
    elif(prob.getInitRBF != [] and unitName == 'RBF'):
        alphaInit, ZInit = prob.getInitRBF
    elif(prob.getInitRBFNorm != [] and unitName == 'RBFNorm'):
        alphaInit, ZInit = prob.getInitRBFNorm
    elif(prob.getInitRBFp != [] and unitName == 'RBFp'):
        alphaInit, ZInit = prob.getInitRBFp
    elif(prob.getInitTanh != [] and unitName == 'tanh'):
        alphaInit, ZInit = prob.getInitTanh
    else:
        raise Exception('Unknown getInitRBF and unit combination')

    if(plotInit):
        # plot
        plt.ion()
        plt.show()

        sampleFun = lambda i, key: prob.sampleNearUT(key, tTimes[i], prob.sampleTargetedLength, FDxGrid)
        pyngplot.doAnimationdD([], jax.jit(lambda x, i:dnn.ufunAZ(x, alphaInit, ZInit)), [], lambda x, i: prob.u0(x), [0], prob.OmegaInit, cmap = prob.colorScheme, sampleFun = sampleFun, sortFun = prob.sortFun, fname = [], key = key, closePlot = 0)
        plt.draw()
        plt.pause(0.001)
        plt.show()

    if(onlyInit): # stop here after truth and initial condition
        input('.')
        quit()


    # solve
    if(schemeName == 'BwdE'):
        alphaStore, ZStore, gradnorm, solverParamStr, storeT, resList, key = timeSGD(ops, dnn, solverName, weightInit, prob.sampleData, nrTSteps, deltat, alphaInit, ZInit, batchSize, miniBatchSize, h, nrIter, tIndx, key)
    elif(schemeName == 'RK23' or schemeName == 'RK45'):
        alphaStore, ZStore, gradnorm, solverParamStr, storeT, resList, key = timeODE(ops, dnn, schemeName, weightInit, prob.sampleData, nrTSteps, deltat, alphaInit, ZInit, batchSize, miniBatchSize, h, nrIter, tIndx, prob.sampleForRes, key)
    else:
        raise Exception('Unknown scheme')

    alphaStore = jnp.asarray(alphaStore)
    ZStore = jnp.asarray(ZStore)
    if(prob.outputNormalize == True):
        U = jax.jit(lambda x, i: dnn.ufunNormalizedAZ(x, alphaStore[:, i], ZStore[:, i].reshape((N,-1))))
    else:
        U = jax.jit(lambda x, i: dnn.ufunAZ(x, alphaStore[:, i], ZStore[:, i].reshape((N,-1))))
    # legacy variables
    Ustore = []
    Ubench = []
    plotGridStore = []
    FDxGridStore = []

    # store
    fname = 'results/' + modeName + '_' + probName + '_' + unitName + '_' + schemeName + "_" + solverName + solverParamStr + "_" + sampleName + "_" + weightInit + '_ZI' + zInitMode + '-' + initSampleName + '-' + "{:.0e}".format(initBatchSize) + '-' + "{:.0e}".format(initNrIter) + '-' + "{:.0e}".format(initH) + '-' + "{:}".format(nrReplicates) + '_N' + "{:}".format(N) + "_nrL" + "{:}".format(nrLayers) + "_dt" + str(deltat) + '_bs' + str(batchSize) + '-' + str(miniBatchSize) + '_nrIter' + str(nrIter) + '_h' + str(h) + postfix

    if(writeToFile):
        print(fname + ".npz")
        jnp.savez(fname + '.npz', U=Ustore, knots=[], Ubenchmark = Ubench, gradnorm=gradnorm, resList = resList, alphaStore=alphaStore, tTimes = tTimes, ZStore = ZStore, probName = probName, unitName = unitName, plotGrid = plotGridStore, evalphi = [], FDxGrid = FDxGridStore, tIndx = tIndx, deltat = deltat, Tend = prob.Tend, storeT = storeT, Omega = prob.Omega, OmegaInit = prob.OmegaInit)

    # plot
    if(plotFinal or writeVideo):
        sampleFun = lambda i, key: prob.sampleNearUT(key, tTimes[i], prob.sampleTargetedLength, FDxGrid)
        anim, key = pyngplot.doAnimationdD([], U, [], Ubenchmark, tTimes, prob.Omega, cmap = prob.colorScheme, sortFun = prob.sortFun, sampleFun = sampleFun, key = key, fname = [], closePlot = 0);

        if(writeVideo):
            anim.save(fname + '.mp4')

        if(plotFinal):
            input('Press key...')
            plt.show()

def getOps(prob, unit, schemeName, modeName):
    if(prob.probName[:3] == "Adv"):
        ops = OpsAdv(prob, unit, schemeName, modeName)
    elif(prob.probName[:8] == "Particle"):
        ops = OpsParticleTrap(prob, unit, schemeName, modeName)
    elif(prob.probName[:3] == "KdV"):
        ops = OpsKdV(prob, unit, schemeName, modeName)
    else:
        raise Exception("Unknown operators")

    return ops

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modeName", help = "mode", choices = ['NG', 'F2'])
    parser.add_argument("probName", help = "problem name")
    parser.add_argument("unitName", help = "unit name", choices=["RBF", "tanh", "RBFp", 'tanhp', 'RBFNorm', 'hat'])
    parser.add_argument("schemeName", help = "name of time integration scheme", choices = ["FwdE", "BwdE", "RK45", "RK23"])
    parser.add_argument("solverName", help = "name of solver", choices = ["adamSGD", "adamSGDMin"])
    parser.add_argument("weightInit", help = "initializing weights at each time step", choices = ["reuse", "zero", "randn"])
    parser.add_argument("sampleName", help = "how to sample data", choices = ["uni", "equi", "gauss", "gauss1", "gauss2", "gauss3", "gauss4", "gauss5"])
    parser.add_argument("N", help = "number of nodes", type = int)
    parser.add_argument("nrLayers", help = "number of hidden layers", type = int)
    parser.add_argument("batchSize", help = "batch size", type = float)
    parser.add_argument("miniBatchSize", help = "mini batch size", type = float)
    parser.add_argument("lr", help = "learning rate", type = float)
    parser.add_argument("deltat", help = "time-step size", type = float)
    parser.add_argument("nrIter", help = "number of SGD iterations", type = float)
    parser.add_argument("initSampleName", help = "sampling for fitting initial condition", choices = ["uni", "equi", "gauss"])
    parser.add_argument("initBatchSize", help = "batch size of fitting initial condition", type = float)
    parser.add_argument("initNrIter", help = "number of SGD iterations when fitting initial condition", type = float)
    parser.add_argument("initLr", help = "learning rate for fitting initial condition", type = float)
    parser.add_argument("zInitMode", help = "mode of fitting initial condition")
    parser.add_argument("nrReplicates", help = "number of replicates", type = int, default = 1)
    parser.add_argument("-i", "--plotInit", help = "yes/no plot initial condition", type = int, choices=[0, 1], default = 0)
    parser.add_argument("-p", "--plotFinal", help = "yes/no plot prediction", type = int, choices=[0, 1], default = 0)
    parser.add_argument("-w", "--writeVideo", help = "yes/no write video", type = int, choices=[0, 1], default = 0)
    parser.add_argument("-o", "--onlyInit", help = "yes/no only compute initial condition and truth and then exit", type = int, choices=[0, 1], default = 0)
    parser.add_argument("-f", "--writeToFile", help = "yes/no write results to npz file", type = int, choices=[0, 1], default = 0)
    parser.add_argument("-r", "--postfix", help = "add this to the filenames", default = '')
    args = parser.parse_args()

    testpyNG(args.modeName, args.probName, args.unitName, args.schemeName, args.solverName, args.weightInit, args.sampleName, args.N, args.nrLayers, int(args.batchSize), int(args.miniBatchSize), args.lr, args.deltat, int(args.nrIter), args.initSampleName, int(args.initBatchSize), int(args.initNrIter), args.initLr, args.zInitMode, args.nrReplicates, args.plotInit, args.plotFinal, args.writeVideo, args.onlyInit, args.writeToFile, args.postfix)
