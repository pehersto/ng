import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
import numpy as np
from functools import partial

class DNN:
    # unitName, number of units per layer and number of hidden layers, number of inputs
    def __init__(self, unitName, N, M, p, Omega):

        curUfunScalarDXDXX = None
        self.unitfunInputDXDXX = None
        self.N = N
        self.M = M
        self.p = p

        if(unitName == 'tanh' or unitName == 'softplus' or unitName == 'relu'):
            if(unitName == 'tanh'):
                self.unitfun = unittanh
                self.unitfunInput = unittanh
            elif(unitName == 'softplus'):
                self.unitfun = unitsoftplus
                self.unitfunInput = unitsoftplus
            elif(unitName == 'relu'):
                self.unitfun = unitrelu
                self.unitfunInput = unitrelu
            else:
                raise Exception("Unknown unit")
            self.initZ = self.initZRnd
            self.knots = lambda Z: []
            self.unitIsPeriodic = 0
            curUfunScalar = ufunScalarHelper
        elif(unitName == 'tanhp'):
            self.unitfun = unittanh
            self.unitfunInput = partial(unittanhpInput, Lp = 2*jnp.pi/(Omega[1, :] - Omega[0, :]))
            self.initZ = self.initZRnd
            self.knots = lambda Z: []
            self.unitIsPeriodic = 1
            curUfunScalar = ufunScalarHelper
        elif(unitName == 'RBFp'):
            self.makePeriodic = lambda x: jnp.mod(x - Omega[0, :], Omega[1, :] - Omega[0, :]) + Omega[0, :]
            self.unitfunInput = partial(unitrbfpInput, Lp = jnp.pi/(Omega[1, :] - Omega[0, :]))
            self.unitfun = unitrbf
            self.knots = lambda Z: [] # jit(lambda Z: jnp.mod(Z[:, p] - Omega[0], Omega[1] - Omega[0]) + Omega[0])
            self.unitIsPeriodic = 1
            self.initZ = self.initZRnd
            self.initZStatic = lambda : self.initZGridPeriodic(Omega)
            curUfunScalar = ufunScalarHelper
        elif(unitName == 'RBF' or unitName == 'RBFNorm'):
            self.unitfunInput = unitrbfInput
            self.unitfunInputNormalized = unitrbfNormalizedInput
            self.unitfunInputDXDXX = unitrbfInputDXDXX
            self.unitfun = unitrbf
            self.initZ = self.initZUni
            self.knots = lambda Z: [] # jit(lambda Z: Z[:, 1:])
            self.unitIsPeriodic = 0
            self.initZStatic = lambda : self.initZGrid(Omega)
            if(unitName == 'RBF'):
                curUfunScalar = ufunScalarHelper
                curUfunScalarDXDXX = ufunScalarDXDXXHelper
            elif(unitName == 'RBFNorm'):
                curUfunScalar = ufunScalarNormalizedWeightsHelper
                curUfunScalarDXDXX = ufunScalarNormalizedWeightsDXDXXHelper
            else:
                raise Exception('Unknown RBF')
        elif(unitName == 'hat'):
            self.unitfunInput = unithat
            self.unitfun = []
            self.initZ = self.initZUni
            self.knots = lambda Z: [] # jit(lambda Z: Z[:, 1:])
            self.unitIsPeriodic = 0
            self.initZStatic = lambda : self.initZHat(None, Omega)
            curUfunScalar = ufunScalarHelper
        else:
            raise Exception('Unknow unit')

        self.ufunDimScalar = lambda *xcHat: self.ufunScalar(jnp.asarray(xcHat[:-1]), xcHat[-1])
        self.ufunScalar = lambda x, cHat: curUfunScalar(N, M, p, self.unitfunInput, self.unitfun, x, cHat)
        self.ufunScalarDXDXX = lambda x, cHat: curUfunScalarDXDXX(N, M, p, self.unitfunInputDXDXX, x, cHat)
        self.ufunScalarAZ = lambda x, alpha, Z: curUfunScalar(N, M, p, self.unitfunInput, self.unitfun, x, jnp.hstack((alpha.reshape((-1,1)), Z)).reshape((-1,)))
        self.ufun = lambda x, cHat: vmap(curUfunScalar, in_axes = (None, None, None, None, None, 0, None), out_axes = 0)(N, M, p, self.unitfunInput, self.unitfun, jnp.atleast_1d(x), cHat)
        self.ufunAZ = jit(lambda x, alpha, Z: vmap(curUfunScalar, in_axes = (None, None, None, None, None, 0, None), out_axes = 0)(N, M, p, self.unitfunInput, self.unitfun, x, jnp.hstack((alpha.reshape((-1,1)), Z)).reshape((-1,))))
        self.ufunNormalizedAZ = lambda x, alpha, Z: 1./jnp.sum(alpha)*vmap(curUfunScalar, in_axes = (None, None, None, None, None, 0, None), out_axes = 0)(N, M, p, self.unitfunInputNormalized, self.unitfun, x, jnp.hstack((alpha.reshape((-1,1)), Z)).reshape((-1,)))
        self.paramShape = (N, p+2+(M-1)*(N+1))

    def getAlpha(self, az):
        return az.reshape((self.N, -1))[:, 0]

    def getZ(self, az):
        return az.reshape((self.N, -1))[:, 1:self.paramShape[1]]

    def evale(self, x, Z):
        N = Z.shape[0]
        M = x.shape[0]
        evale = np.zeros((M, N))
        for i in range(N):
            alphaTmp = np.zeros((N,))
            alphaTmp[i] = 1.
            evale[:, i] = self.ufunAZ(x, alphaTmp, Z)
        return jnp.asarray(evale)

    def initZGrid(self, Omega):
        h = (Omega[1] - Omega[0])/(self.N + 1) # distance between grid points
        valueAtHHalf = 1./2.
        a = jnp.sqrt(-2*jnp.log(valueAtHHalf)/((h/2.)**2))
        Z = jnp.array(jnp.hstack((a*jnp.ones((self.N, 1)), jnp.linspace(Omega[0], Omega[1], self.N).reshape((self.N, 1)))))
        return Z

    def initZHat(self, key, Omega):
        h = (Omega[1] - Omega[0])/(self.N - 1)
        Z = jnp.asarray(jnp.hstack((h*jnp.ones((self.N, 1)), jnp.linspace(Omega[0], Omega[1], self.N).reshape((self.N, 1)))))
        return Z

    def initZGridPeriodic(self, Omega):
        # bandwidth selection
        h = (Omega[1] - Omega[0])/(self.N + 1) # distance between grid points
        valueAtHHalf = 1./2.
        a = jnp.sqrt(-jnp.log(valueAtHHalf)/jnp.sin(jnp.pi/(Omega[1] - Omega[0])*(h/2.))**2)

        Z = jnp.array(jnp.hstack((a*jnp.ones((self.N, 1)), jnp.linspace(Omega[0], Omega[1], self.N+1)[:self.N].reshape((self.N, 1)))))
        return Z

    def initZRnd(self, key, Omega):
        key, subkey = jax.random.split(key)
        Z = jax.random.normal(subkey, (self.paramShape[0], self.paramShape[1]-1))
        return Z, key

    def initZUni(self, key, Omega):
        key, subkey = jax.random.split(key)
        key, subkey2 = jax.random.split(key)
        Z = jnp.hstack((4*jax.random.normal(subkey, (self.paramShape[0], 1)), jax.random.uniform(subkey2, (self.paramShape[0], self.paramShape[1]-2))*(Omega[1, :] - Omega[0, :]) + Omega[0, :]))
        return Z, key

    def getInitAZ(self, key, Omega):
        key, subkey = jax.random.split(key)
        alphaInit = jax.random.normal(subkey, shape=(self.paramShape[0],))
        ZInit, key = self.initZ(key, Omega)
        return alphaInit, ZInit, key

def ufunScalarHelper(N, M, p, unitfunInput, unitfun, x, cHat):
    print('compile ufun')
    cHat = cHat.reshape((N,-1))
    y = unitfunInput(x, cHat[:, 1:p+2])
    for i in jnp.arange(1, M):
        y = unitfun(y, cHat[:, p+2+(i-1)*(N+1):p+2+i*(N+1)])
    y = jnp.vdot(cHat[:, 0], y).reshape(()) # make sure to return a scalar
    return y

def ufunScalarDXDXXHelper(N, M, p, unitfunInputDXDXX, x, cHat):
    print('compile ufunDXDXX')
    cHat = cHat.reshape((N,-1))
    yx, yxx = unitfunInputDXDXX(x, cHat[:, 1:p+2])
    return jnp.dot(cHat[:, 0], yx).reshape((p,)), jnp.vdot(cHat[:, 0], yxx).reshape(())

def ufunScalarNormalizedWeightsDXDXXHelper(N, M, p, unitfunInputDXDXX, x, cHat):
    print('compile ufunDXDXX')
    cHat = cHat.reshape((N,-1))
    yx, yxx = unitfunInputDXDXX(x, cHat[:, 1:p+2])
    return jnp.dot(cHat[:, 0]**2, yx).reshape((p,)), jnp.vdot(cHat[:, 0]**2, yxx).reshape(())

def ufunScalarNormalizedWeightsHelper(N, M, p, unitfunInput, unitfun, x, cHat):
    print('compile ufun with normalized weights')
    cHat = cHat.reshape((N,-1))
    y = unitfunInput(x, cHat[:, 1:p+2])
    for i in jnp.arange(1, M):
        y = unitfun(y, cHat[:, p+2+(i-1)*(N+1):p+2+i*(N+1)])
    y = jnp.vdot(cHat[:, 0]**2, y).reshape(()) # make sure to return a scalar
    return y

@jit
def unittanh(x, Z):
    print('compile unit tanh')
    return jnp.tanh(jnp.dot(Z[:, :-1], x.reshape((-1,1))) + Z[:, -1].reshape((-1,1)))

@jit
def unittanhpInput(x, Z, Lp = 1.0):
    print('compile unit tanhp')
    # WARNING: We hard-code (for speed) that there is only one input (dim(x) = 1)
    #print(Z.shape, jnp.sin(Lp.reshape((1,-1))*(x.reshape((1,-1)) + Z[:, -1].reshape((-1,1)))).shape)
    return jnp.tanh(jnp.sum(jnp.multiply(Z[:, :-1], jnp.sin(Lp.reshape((1,-1))*(x.reshape((1,-1)) + Z[:, -1].reshape((-1,1))))), axis = 1)).reshape((Z.shape[0],1))

@jit
def unitsoftplus(x, Z):
    print('compile unit softplus')
    return jnp.log(1.0 + jnp.exp(jnp.dot(Z[:, :-1], x) + Z[:, -1].reshape((-1,1))))

@jit
def unitrelu(x, Z):
    print('compile unit relu')
    return jnp.maximum(0, jnp.dot(Z[:, :-1], x) + Z[:, -1].reshape((-1,1)))

@jit
def unitrbf(x, Z):
    print('compile unit rbf')
    return jnp.exp(-jnp.square(jnp.dot(Z[:, :-1], x) - Z[:, -1].reshape((-1,1))))

@jit
def unitrbfInput(x, Z):
    print('compile unit rbfInput')
    return jnp.exp(-0.5*jnp.sum(jnp.multiply(x - Z[:, 1:], jnp.multiply(jnp.square(Z[:, 0]).reshape((-1,1)), x - Z[:, 1:])), axis = 1))

@jit
def unitrbfInputDXDXX(x, Z):
    print('compile unit rbfInputDXDXX')
    y = jnp.exp(-0.5*jnp.sum(jnp.multiply(x - Z[:, 1:], jnp.multiply(jnp.square(Z[:, 0]).reshape((-1,1)), x - Z[:, 1:])), axis = 1))
    dx = jnp.multiply(-y.reshape((-1,1))*jnp.square(Z[:, 0]).reshape((-1, 1)), x - Z[:, 1:])
    dxx = jnp.multiply(y.reshape((-1,))*jnp.square(Z[:, 0]).reshape((-1,)), jnp.sum(jnp.square(Z[:, 0].reshape((-1, 1)))*jnp.square(x - Z[:, 1:]) - 1.0, axis = 1).reshape((-1,)))
    return dx, dxx

@jit
def unitrbfNormalizedInput(x, Z):
    print('compile unit rbfNormalizedInput')
    return jnp.multiply(1./jnp.sqrt((2*jnp.pi*jnp.square(1./Z[:, 0]))**x.size), jnp.exp(-0.5*jnp.sum(jnp.multiply(x - Z[:, 1:], jnp.multiply(jnp.square(Z[:, 0]).reshape((-1,1)), x - Z[:, 1:])), axis = 1)))

@jit
def unitrbfpInput(x, Z, Lp = 1.0):
    print('compile unit rbf periodic')
    return jnp.exp(-0.5*jnp.sum(jnp.multiply(jnp.sin(Lp*(x - Z[:, 1:])), jnp.multiply(jnp.square(Z[:, 0]).reshape((-1,1)), jnp.sin(Lp*(x - Z[:, 1:])))), axis = 1))

@jit
def unithat(x, Z):
    print('compile unit hat')
    assert(Z.shape[1] == 2) # only support 1D
    return jnp.maximum(1 - jnp.abs((x - Z[:, 1])/Z[:, 0]), 0)
