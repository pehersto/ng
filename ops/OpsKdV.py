import jax
from jax import grad, jit, vmap, value_and_grad, jvp
import jax.numpy as jnp
from functools import partial

class OpsKdV:
    def __init__(self, prob, dnn, scheme, modeName):
        self.prob = prob # problem
        self.dnn = dnn
        self.modeName = modeName

        # check if dnn satisifies boundary conditions
        if(self.prob.OmegaPeriodic == 1 and self.dnn.unitIsPeriodic == 1):
            print("No boundary penality because units satisfy boundary conditions")
            self.bc = lambda xdfun: 0.
        else:
            print("Enforcing boundary conditions via penalty")
            self.bc = self.prob.bc

        if(scheme == "RK23" or scheme == "RK45"):
            if(modeName == 'NG'):
                self.rhsJ = self.rhsJF1
            elif(modeName == 'F2'):
                self.rhsJ = self.rhsJF2
            else:
                raise Exception('Not implemented')
        else:
            raise Exception("not implemented")

    def rhsJF1(self, ZInit, x, alphaZ, t):
        return KdVRHS(self.dnn.ufunScalar, self.dnn.ufun, self.bc, self.prob.nu(t), x.reshape((-1,)), alphaZ, t)

    def rhsJF2(self, ZInit, x, alpha, t):
        return KdVRHSF2(self.dnn.ufunScalarAZ, self.dnn.ufunAZ, self.bc, self.prob.nu(t), x.reshape((-1,)), alpha, ZInit, t)

@partial(jit, static_argnums=(0,1,2,))
def KdVRHS(ufunScalar, ufun, bc, nu, x, alphaZ, t):
    print('compile rhs')

    dx = jax.vmap(grad(ufunScalar, argnums = 0), in_axes = (0, None), out_axes = 0)(x, alphaZ)
    dxxx = jax.vmap(grad(grad(grad(ufunScalar, argnums = 0), argnums = 0), argnums = 0), in_axes = (0, None), out_axes = 0)(x, alphaZ)
    Jac = jax.jacfwd(lambda az: ufun(x, az))(alphaZ)
    K = jnp.dot(Jac.T, Jac)
    u = ufun(x, alphaZ)
    f = jnp.dot(Jac.T, nu*jnp.multiply(u, dx) + dxxx)

    J = jnp.linalg.lstsq(K, -f)[0]

    return J

@partial(jit, static_argnums=(0,1,2,))
def KdVRHSF2(ufunScalar, ufun, bc, nu, x, alpha, ZInit, t):
    print('compile rhs F2')

    dx = jax.vmap(grad(ufunScalar, argnums = 0), in_axes = (0, None, None), out_axes = 0)(x, alpha, ZInit)
    dxxx = jax.vmap(grad(grad(grad(ufunScalar, argnums = 0), argnums = 0), argnums = 0), in_axes = (0, None, None), out_axes = 0)(x, alpha, ZInit)
    Jac = jax.jacfwd(lambda a: ufun(x, a, ZInit))(alpha)
    K = jnp.dot(Jac.T, Jac)
    u = ufun(x, alpha, ZInit)
    f = jnp.dot(Jac.T, nu*jnp.multiply(u, dx) + dxxx)

    J = jnp.linalg.lstsq(K, -f)[0]

    return J
