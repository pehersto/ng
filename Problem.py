import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
from jax import random
import numpy as np
from scipy import special
import scipy
from jax.ops import index, index_add, index_update
from matplotlib import cm
from functools import partial
import math

from solvers import exactKdV, fpesolver
from misc import pyngtools

from datetime import datetime

# DEBUG
import matplotlib.pyplot as plt

class Problem:
    def __init__(self, probName, sampleName, N, nrLayers, deltat = -1):
        self.N = N
        self.nrLayers = nrLayers
        self.probName = probName
        self.sampleName = sampleName
        self.deltat = deltat
        self.storeSol = 0
        self.sampleTargetedLength = 20000
        self.getInitRBF = []
        self.getInitRBFp = []
        self.getInitTanh = []
        self.outputNormalize = False
        self.sampleForRes = []

        if(probName == "KdVTwoSol"):
            self.setupKdVTwoSol()
        elif(probName[:12] == 'AdvTimeCoeff'):
            self.setupAdvTimeCoeff(int(probName[12:]))
        elif(probName[:8] == 'Particle'):
            self.setupParticleTrapTime(int(probName[8:]))
        else:
            raise Exception('unknown probname', probName)

        if(sampleName == "uni"):
            self.setupUniformSampling()
        elif(sampleName == "equi"):
            self.setupEquidistantSampling()
        elif(sampleName == "gauss1"):
            self.setupGaussianSamplingUnitRBF(1)
        elif(sampleName == "gauss" or sampleName == "gauss2"):
            self.setupGaussianSamplingUnitRBF(2)
        elif(sampleName == "gauss3"):
            self.setupGaussianSamplingUnitRBF(3)
        elif(sampleName == "gauss4"):
            self.setupGaussianSamplingUnitRBF(4)
        elif(sampleName == "gauss5"):
            self.setupGaussianSamplingUnitRBF(5)
        else:
            raise Exception("unknown sampling scheme")

    def setupAdvTimeCoeff(self, d = 5):
        '''Advection with time-varying coefficient'''

        self.OmegaPeriodic = 0 # periodic domain
        self.dim = d
        self.Omega = jnp.array([self.dim*[0], self.dim*[15.0]])
        self.OmegaInit = jnp.array([self.dim*[0], self.dim*[2.5]])
        self.maxDiffDegree = 1 # maximal degree of derivative in x
        self.Tend = 1.8
        speedFac = jnp.asarray(list(range(1, self.dim+1)));
        speedFacVar = 2*(1 + jnp.linspace(0., 1., self.dim))
        self.nu = lambda x, t: speedFac*jnp.sin(speedFacVar*jnp.pi*t) + 1.25*speedFac # transport coefficient
        self.mu = lambda x, t: 0. # diffusion coefficient
        self.nuInt = lambda T: 1./(jnp.pi*speedFacVar)*(speedFac - speedFac*jnp.cos(speedFacVar*jnp.pi*T)) + 1.25*speedFac*T

        tseed = datetime.now().timestamp()
        key = jax.random.PRNGKey(int(tseed*100))
        self.plotGridInit, key = sampleUniformHelper(self.OmegaInit, 10000000, key)
        self.plotGrid, key = sampleUniformHelper(self.Omega, 10000000, key)

        mmmOne = jnp.asarray(self.dim*[1.1])
        mmmTwo = jnp.asarray([(1.5 - z/(self.dim+1.0)*(-1)**z)*0.75 for z in range(1,self.dim+1)])
        sigmaOne = jnp.asarray([0.005*(2*i) for i in range(1,self.dim+1)])
        sigmaTwo = jnp.asarray([0.005*(self.dim - i + 1) for i in range(0,self.dim)])
        print(mmmOne, mmmTwo, sigmaOne, sigmaTwo)
        self.u0 = jax.jit(lambda x: pyngtools.mvnpdf(x, mean = mmmOne, covdiag = sigmaOne).reshape((-1,)) + pyngtools.mvnpdf(x, mean = mmmTwo, covdiag = sigmaTwo).reshape((-1,)))

        self.uT = lambda x, t: jax.vmap(lambda ti: self.u0(x.reshape((-1,d)) - self.nuInt(ti).reshape((1, d))), out_axes = 1)(t)

        self.getInitRBFp = []
        self.getInitRBF = []
        self.getInitRBFNorm = []

        self.sampleNearUT = lambda key, t, Nx, FDxGrid: 2*[pyngtools.sampleGaussianMixture(key, Nx, [mmmOne + self.nuInt(t), mmmTwo + self.nuInt(t)], [5*sigmaOne, 5*sigmaTwo])[0]] + [pyngtools.sampleGaussianMixture(key, 10, [mmmOne + self.nuInt(t), mmmTwo + self.nuInt(t)], [5*sigmaOne, 5*sigmaTwo])[1]]

        self.bc = lambda sol: 0

        self.colorScheme = cm.gnuplot2_r
        self.sortFun = lambda x: x

    def setupKdVTwoSol(self):
        '''KdV with two solitons'''

        self.OmegaPeriodic = 1 # periodic domain
        self.dim = 1
        self.Omega = jnp.array([-20., 40.]).reshape((2, 1))
        self.OmegaInit = self.Omega
        self.maxDiffDegree = 3 # maximal degree of derivative in x
        self.Tend = 4.0
        self.nu = lambda t: 6. # transport coefficient

        self.u0 = lambda x: exactKdV.exactKdVTwoSol(x, 0)
        self.uT = lambda x,t: exactKdV.exactKdVTwoSol(x, t)

        # boundary condition (only used if unit doesn't satisfy boundary condition)
        self.bc = lambda sol: jnp.sqrt(jnp.sum(jnp.square(sol(jnp.atleast_2d(self.Omega[0, 0])) - sol(jnp.atleast_2d(self.Omega[1, 0])))))
        self.balanceWeight = 1. # for DGM

        self.plotGrid = jax.numpy.linspace(self.Omega[0, 0], self.Omega[1, 0], num = 2048).reshape((-1,1));
        self.plotGridInit = jax.numpy.linspace(self.OmegaInit[0, 0], self.OmegaInit[1, 0], num = 20480).reshape((-1,1));
        self.sampleTargetedLength = self.plotGrid.shape[0]

        self.getInitRBFp = []
        self.getInitRBF = []
        self.getInitRBFNorm = []

        self.colorScheme = cm.gnuplot2_r
        self.sortFun = lambda x: x

        self.storeSol = 1

        self.sampleNearUT = lambda key, t, Nx, FDxGrid: 2*[self.plotGrid[:Nx, :]] + [key]

    def setupParticleTrapTime(self, d):
        '''Particles in time-varying harmonic trap'''

        self.OmegaPeriodic = 0 # periodic domain
        self.dim = d
        self.Omega = jnp.array([self.dim*[0.], self.dim*[5.0]])
        self.OmegaInit = jnp.array([self.dim*[0.], self.dim*[5.0]])
        self.maxDiffDegree = 1 # maximal degree of derivative in x
        self.Tend = 8.0

        self.nu = lambda t: 1.25*(jnp.sin(jnp.pi*t) + 1.5) # a
        self.alpha = 0.25 # alpha
        self.beta = 100. # beta
        mmmOne = jnp.linspace(0.9, 3, self.dim)
        sigmaOne = jnp.asarray(self.dim*[0.1])

        self.outputNormalize = True

        tseed = datetime.now().timestamp()
        key = jax.random.PRNGKey(int(tseed*100))
        self.plotGridInit, key = sampleUniformHelper(self.OmegaInit, 100000, key)
        self.plotGrid, key = sampleUniformHelper(self.Omega, 100000, key)

        self.u0 = jax.jit(lambda x: pyngtools.mvnpdf(x, mean = mmmOne, covdiag = sigmaOne).reshape((-1,)))

        self.uT = lambda x, t: pyngtools.mvnpdfFull(x, meanAndCovList = fpesolver.fpesolverUT(mmmOne, t, self.nu, self.alpha, self.beta, cov0 = jnp.diag(sigmaOne)))
        self.uMeanCov = lambda t: fpesolver.fpesolverUT(mmmOne, t, self.nu, self.alpha, self.beta, cov0 = jnp.diag(sigmaOne))

        self.sampleNearUT = lambda key, t, Nx, FDxGrid: 2*[pyngtools.samplemvn(key, Nx, meanAndCov = fpesolver.fpesolverUT(mmmOne, t, self.nu, self.alpha, self.beta, cov0 = jnp.diag(sigmaOne)), facScale = 20)[0]] + [pyngtools.samplemvn(key, 10, meanAndCov = fpesolver.fpesolverUT(mmmOne, t, self.nu, self.alpha, self.beta, cov0 = jnp.diag(sigmaOne)), facScale = 20)[1]]

        self.bc = lambda sol: 0

        self.getInitRBF = 1./jnp.sqrt((2*jnp.pi)**self.dim*jnp.prod(sigmaOne))*1.0/self.N*jnp.ones((self.N,)), jnp.hstack((1./jnp.sqrt(sigmaOne[0])*jnp.ones((self.N,1)), jnp.tile(mmmOne, (self.N, 1))))
        self.getInitRBFNorm = jnp.sqrt(1./jnp.sqrt((2*jnp.pi)**self.dim*jnp.prod(sigmaOne))*1.0/self.N*jnp.ones((self.N,))), jnp.hstack((1./jnp.sqrt(sigmaOne[0])*jnp.ones((self.N,1)), jnp.tile(mmmOne, (self.N, 1))))

        self.colorScheme = cm.gnuplot2_r
        self.sortFun = lambda x: x

    def setupUniformSampling(self):
        """Uniform distribution in Omega"""
        self.sampleData = lambda Nx, key, phi, alpha, Z, knots: sampleUniformHelper(self.Omega, Nx, key)
        self.sampleDataInit = lambda Nx, key, phi, alpha, Z, knots: sampleUniformHelperFocus(self.Omega, self.OmegaInit, Nx, key)

    def setupEquidistantSampling(self):
        """Equidistant grid in Omega"""
        self.sampleData = lambda Nx, key, phi, alpha, Z, knots: sampleEquidistantHelper(self.Omega, Nx, key)
        self.sampleDataInit = lambda Nx, key, phi, alpha, Z, knots: sampleEquidistantHelper(self.OmegaInit, Nx, key)

    def setupGaussianSampling(self):
        """Fit Gaussian to each unit and sample from corresponding mixture"""
        error('dont use anymore')
        self.sampleData = lambda Nx, key, phi, alpha, Z, knots: sampleGaussHelper(self.Omega, Nx, key, phi, alpha, Z, knots)
        self.sampleDataInit = self.sampleData

    def setupGaussianSamplingUnitRBF(self, factorSigma = 2):
        """Fit Gaussian to each unit and sample from corresponding mixture"""
        self.sampleData = lambda Nx, key, phi, alpha, Z, knots: sampleGaussUnitRBFHelper(Nx, key, Z, factorSigma)
        self.sampleDataInit = self.sampleData


def sampleUniformHelper(Omega, Nx, key):
    key, subkey = jax.random.split(key)
    return jax.random.uniform(subkey, (Nx, Omega.shape[1]), minval = 0., maxval = 1.)*(Omega[1, :] - Omega[0, :]) + Omega[0, :], key

def sampleUniformHelperFocus(Omega, OmegaInit, Nx, key):
    key, subkey = jax.random.split(key)
    key, ssubkey = jax.random.split(key)
    return jax.numpy.vstack([jax.random.uniform(subkey, (int(Nx/4), Omega.shape[1]), minval = 0., maxval = 1.)*(Omega[1, :] - Omega[0, :]) + Omega[0, :], jax.random.uniform(ssubkey, (int(3*Nx/4), OmegaInit.shape[1]), minval = 0., maxval = 1.)*(OmegaInit[1, :] - OmegaInit[0, :]) + OmegaInit[0, :]]), key

def sampleEquidistantHelper(Omega, Nx, key):
    # key, subkey = jax.random.split(key)
    return jax.numpy.linspace(Omega[0], Omega[1], num = Nx), key

#@partial(jit, static_argnums=(0,))
def sampleGaussUnitRBFHelper(Nx, key, Z, factorSigma = 2.0):
    N = Z.shape[0]
    d = Z.shape[1]-1
    NStep = int(math.ceil(Nx/N)) # sample from each Gaussian Nx/N points
    key, subkey = jax.random.split(key)
    xSamples = (Z[:, 1:].reshape((N, 1, d)) + jnp.multiply(factorSigma/jnp.abs(Z[:, 0]).reshape((N, 1, 1)), jax.random.normal(subkey, shape = (N, NStep, d)))).reshape((N*NStep, d))

    # plt.ion()
    # plt.show()
    # fig, ax = plt.subplots()
    # ax.plot(jnp.linspace(Omega[0], Omega[1], Nx), evale + jnp.dot(jnp.ones(evale.shape), jnp.diag(jnp.linspace(1, N, N))), '-')
    # ax.plot(xSamples, jnp.dot(jnp.ones((NStep, N)), jnp.diag(jnp.linspace(1, N, N))), '-k')
    #
    # plt.draw()
    # plt.pause(0.001)
    # # plt.show()
    # input('..')

    xSamples = xSamples[:Nx, :]
    return xSamples, key

def sampleGaussHelper(Omega, Nx, key, phi, alpha, Z, knots):
    N = alpha.size
    NStep = int(jnp.ceil(Nx/N)) # sample from each Gaussian Nx/N points
    xSamples = jnp.zeros((NStep*N,))
    bw = (Omega[1] - Omega[0])/2. # bandwidth

    for i in range(N):
        y = jnp.linspace(knots[i]-bw, knots[i]+bw, num = Nx)
        u = phi(y, Z[i, :].reshape((1,-1))).reshape((-1,))
        normalizingConstant = jnp.divide(jnp.square(u), jnp.mean(jnp.square(u)))
        meanU = jnp.mean(jnp.multiply(y, normalizingConstant))
        sigmaU = 2*jnp.sqrt(jnp.mean((y - meanU)**2*normalizingConstant)) # add a 2 for larger variance
        key, subkey = jax.random.split(key)
        xSamples = index_update(xSamples, index[i*NStep:(i+1)*NStep], meanU + sigmaU*jax.random.normal(subkey, shape = (NStep,)))

    xSamples = xSamples[:Nx]

    return xSamples, key
